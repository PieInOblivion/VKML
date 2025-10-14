use std::collections::HashSet;
use std::sync::OnceLock;

use crate::compute::compute_manager::ComputeManager;
use crate::tensor::tensor::DeviceId;
use crate::tensor_graph::tensor_graph::OperationId;
use crate::utils::error::VKMLError;
use vulkanalia::vk;

pub type ChunkId = usize;

pub struct ExecutionChunk {
    pub device: DeviceId,
    pub operation_layers: Vec<Vec<OperationId>>,
    pub predecessors: Vec<ChunkId>,
    pub dependents: Vec<ChunkId>,
    pub initial_dep_count: usize,
    pub is_output: bool,
    pub needs_host_wait: bool,
    pub command_buffer: OnceLock<vk::CommandBuffer>,
}

pub struct ExecutionPlan {
    pub chunks: Vec<ExecutionChunk>,
    pub output_chunks: Vec<ChunkId>,
    pub root_chunks: Vec<ChunkId>,
}

impl ExecutionPlan {
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }
}

fn organise_chain_into_layers(
    chain: &[OperationId],
    predecessors: &[Vec<OperationId>],
    successors: &[Vec<OperationId>],
    op_count: usize,
) -> Vec<Vec<OperationId>> {
    let mut in_degree: Vec<usize> = vec![0; op_count];
    let chain_set: HashSet<OperationId> = chain.iter().copied().collect();

    for &op in chain {
        for &pred in &predecessors[op] {
            if chain_set.contains(&pred) {
                in_degree[op] += 1;
            }
        }
    }

    let mut layers: Vec<Vec<OperationId>> = Vec::new();
    let mut current_layer: Vec<OperationId> = chain
        .iter()
        .copied()
        .filter(|&op| in_degree[op] == 0)
        .collect();

    while !current_layer.is_empty() {
        layers.push(current_layer.clone());

        let mut next_layer: Vec<OperationId> = Vec::new();
        for &op in &current_layer {
            for &succ in &successors[op] {
                if !chain_set.contains(&succ) {
                    continue;
                }
                in_degree[succ] = in_degree[succ].saturating_sub(1);
                if in_degree[succ] == 0 {
                    next_layer.push(succ);
                }
            }
        }
        current_layer = next_layer;
    }

    layers
}

pub fn create_execution_plan(compute_manager: &ComputeManager) -> Result<ExecutionPlan, VKMLError> {
    let tensor_graph = &compute_manager.tensor_graph;
    if tensor_graph.operations.is_empty() {
        return Err(VKMLError::Generic(
            "Scheduler cannot execute an empty graph".into(),
        ));
    }

    let op_count = tensor_graph.operations.len();
    let tensor_count = tensor_graph.tensor_descs.len();

    let dep_graph = compute_manager.dependency_graph();
    let predecessors = &dep_graph.predecessors;
    let successors = &dep_graph.successors;
    let topo_order = &dep_graph.topological_order;

    let op_devices: Vec<DeviceId> = (0..op_count)
        .map(|op_id| {
            let op = &tensor_graph.operations[op_id];
            let tensor_id = op
                .get_output_tensor_ids()
                .first()
                .copied()
                .or_else(|| op.get_input_tensor_ids().first().copied())
                .expect("Operation must reference at least one tensor");
            compute_manager.tensor_read(tensor_id).device.clone()
        })
        .collect();

    let mut chunks: Vec<ExecutionChunk> = Vec::new();
    let mut operation_to_chunk: Vec<ChunkId> = vec![usize::MAX; op_count];
    let mut chain_marks: Vec<u32> = vec![0; op_count];
    let mut chain_mark_value: u32 = 1;

    for &op in topo_order {
        if operation_to_chunk[op] != usize::MAX {
            continue;
        }

        let device = op_devices[op].clone();
        let mut chain: Vec<OperationId> = Vec::new();

        chain_mark_value = chain_mark_value.wrapping_add(1);
        if chain_mark_value == 0 {
            chain_marks.fill(0);
            chain_mark_value = 1;
        }

        // Expand the chain breadth-first so we can include multiple ready successors
        // on the same device (reduces number of chunks created).
        chain_marks[op] = chain_mark_value;
        chain.push(op);
        let mut check_idx: usize = 0;
        while check_idx < chain.len() {
            let cur = chain[check_idx];
            for &succ in &successors[cur] {
                if operation_to_chunk[succ] != usize::MAX {
                    continue;
                }
                if chain_marks[succ] == chain_mark_value {
                    continue;
                }
                if op_devices[succ] != device {
                    continue;
                }

                // first, eagerly include any unassigned zero-dependency predecessors
                // of this successor on the same device. This reduces chunk fragmentation
                // for operations like constant reshapes that have no data dependencies
                for &pred in &predecessors[succ] {
                    if operation_to_chunk[pred] == usize::MAX
                        && chain_marks[pred] != chain_mark_value
                        && predecessors[pred].is_empty()
                        && op_devices[pred] == device
                    {
                        chain_marks[pred] = chain_mark_value;
                        chain.push(pred);
                    }
                }

                // ensure all predecessors are either already assigned to a chunk
                // or are part of this chain (marked)
                if !predecessors[succ].iter().all(|&pred| {
                    operation_to_chunk[pred] != usize::MAX || chain_marks[pred] == chain_mark_value
                }) {
                    continue;
                }

                chain_marks[succ] = chain_mark_value;
                chain.push(succ);
            }
            check_idx += 1;
        }

        let chunk_id = chunks.len();
        for &op_id in &chain {
            operation_to_chunk[op_id] = chunk_id;
        }

        let layers = organise_chain_into_layers(&chain, predecessors, successors, op_count);

        chunks.push(ExecutionChunk {
            device,
            operation_layers: layers,
            predecessors: Vec::new(),
            dependents: Vec::new(),
            initial_dep_count: 0,
            is_output: false,
            needs_host_wait: false,
            command_buffer: OnceLock::new(),
        });
    }

    let chunk_count = chunks.len();
    let mut chunk_predecessors: Vec<Vec<ChunkId>> = vec![Vec::new(); chunk_count];

    for chunk_idx in 0..chunk_count {
        for layer in &chunks[chunk_idx].operation_layers {
            for &op in layer {
                for &pred_op in &predecessors[op] {
                    let pred_chunk = operation_to_chunk[pred_op];
                    if pred_chunk != chunk_idx {
                        chunk_predecessors[chunk_idx].push(pred_chunk);
                    }
                }
            }
        }
        chunk_predecessors[chunk_idx].sort_unstable();
        chunk_predecessors[chunk_idx].dedup();
    }

    // build reverse dependencies (dependents) and populate chunk fields
    let mut chunk_dependents: Vec<Vec<ChunkId>> = vec![Vec::new(); chunk_count];
    for (chunk_idx, preds) in chunk_predecessors.iter().enumerate() {
        for &pred in preds {
            chunk_dependents[pred].push(chunk_idx);
        }
    }

    let mut root_chunks: Vec<ChunkId> = Vec::new();
    for chunk_idx in 0..chunk_count {
        let preds = std::mem::take(&mut chunk_predecessors[chunk_idx]);
        let dependents = std::mem::take(&mut chunk_dependents[chunk_idx]);

        chunks[chunk_idx].initial_dep_count = preds.len();
        chunks[chunk_idx].predecessors = preds;
        chunks[chunk_idx].dependents = dependents;

        if chunks[chunk_idx].initial_dep_count == 0 {
            root_chunks.push(chunk_idx);
        }
    }

    let mut output_tensor_flags = vec![false; tensor_count];
    for &tid in tensor_graph.get_output_tensor_ids() {
        if tid < tensor_count {
            output_tensor_flags[tid] = true;
        }
    }
    let mut output_chunks: Vec<ChunkId> = Vec::new();

    for (chunk_idx, chunk) in chunks.iter_mut().enumerate() {
        let mut is_output = false;
        'outer: for layer in &chunk.operation_layers {
            for &op_id in layer {
                let op = &tensor_graph.operations[op_id];
                if op
                    .get_output_tensor_ids()
                    .iter()
                    .any(|&tid| tid < output_tensor_flags.len() && output_tensor_flags[tid])
                {
                    is_output = true;
                    break 'outer;
                }
            }
        }
        chunk.is_output = is_output;
        if is_output {
            output_chunks.push(chunk_idx);
        }
    }

    if output_chunks.is_empty() {
        for idx in 0..chunk_count {
            chunks[idx].is_output = true;
            output_chunks.push(idx);
        }
    }

    if root_chunks.is_empty() {
        return Err(VKMLError::Generic(
            "Execution plan contains no root chunks".into(),
        ));
    }

    for chunk_idx in 0..chunk_count {
        let needs_wait = match chunks[chunk_idx].device {
            DeviceId::Gpu(gpu_idx) => {
                if chunks[chunk_idx].is_output {
                    true
                } else {
                    chunks[chunk_idx]
                        .dependents
                        .iter()
                        .any(|&dep| match chunks[dep].device {
                            DeviceId::Gpu(dep_gpu) => dep_gpu != gpu_idx,
                            DeviceId::Cpu => true,
                        })
                }
            }
            DeviceId::Cpu => false,
        };
        chunks[chunk_idx].needs_host_wait = needs_wait;
    }

    Ok(ExecutionPlan {
        chunks,
        output_chunks,
        root_chunks,
    })
}
