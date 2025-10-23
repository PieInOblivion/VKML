use std::collections::HashSet;
use std::sync::OnceLock;

use crate::compute::compute_manager::ComputeManager;
use crate::tensor::DeviceId;
use crate::tensor_graph::OperationId;
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
        return Err(VKMLError::GraphScheduler(
            "Scheduler cannot execute an empty graph".into(),
        ));
    }

    let op_count = tensor_graph.operations.len();
    let tensor_count = tensor_graph.tensor_descs.len();
    let gpu_count = compute_manager.gpu_count();
    let cpu_slot = gpu_count;

    let dep_graph = compute_manager.dependency_graph();
    let predecessors = &dep_graph.predecessors;
    let successors = &dep_graph.successors;
    let topo_order = &dep_graph.topological_order;

    let mut chunk_devices: Vec<DeviceId> = Vec::new();
    let mut chunk_operations: Vec<Vec<OperationId>> = Vec::new();
    let mut op_to_chunk: Vec<ChunkId> = vec![usize::MAX; op_count];
    let mut active_chunk_per_slot: Vec<Option<ChunkId>> = vec![None; gpu_count + 1];

    for &op in topo_order {
        // By default use the tensor's device, but if the instruction requires CPU execution
        // force the op onto the CPU slot.
        let op_ref = &tensor_graph.operations[op];
        let mut device = {
            let tensor_id = op_ref
                .get_output_tensor_ids()
                .first()
                .copied()
                .or_else(|| op_ref.get_input_tensor_ids().first().copied())
                .expect("Operation must reference at least one tensor");
            compute_manager.tensor_read(tensor_id).device.clone()
        };

        if op_ref.must_execute_on_cpu() {
            device = DeviceId::Cpu;
        }

        let slot = match device {
            DeviceId::Gpu(idx) => idx,
            DeviceId::Cpu => cpu_slot,
        };

        if slot >= active_chunk_per_slot.len() {
            active_chunk_per_slot.resize(slot + 1, None);
        }

        let reuse_chunk = active_chunk_per_slot[slot].and_then(|chunk_id| {
            let all_local = predecessors[op]
                .iter()
                .all(|&pred| op_to_chunk[pred] == chunk_id);
            if all_local { Some(chunk_id) } else { None }
        });

        let chunk_id = match reuse_chunk {
            Some(id) => id,
            None => {
                let new_id = chunk_operations.len();
                chunk_operations.push(Vec::new());
                chunk_devices.push(device.clone());
                active_chunk_per_slot[slot] = Some(new_id);
                new_id
            }
        };

        chunk_operations[chunk_id].push(op);
        op_to_chunk[op] = chunk_id;
    }

    let mut chunks: Vec<ExecutionChunk> = chunk_operations
        .iter()
        .enumerate()
        .map(|(idx, ops)| {
            let layers = organise_chain_into_layers(ops, predecessors, successors, op_count);
            ExecutionChunk {
                device: chunk_devices[idx].clone(),
                operation_layers: layers,
                predecessors: Vec::new(),
                dependents: Vec::new(),
                initial_dep_count: 0,
                is_output: false,
                needs_host_wait: false,
                command_buffer: OnceLock::new(),
            }
        })
        .collect();

    let chunk_count = chunks.len();
    let mut chunk_predecessors: Vec<Vec<ChunkId>> = vec![Vec::new(); chunk_count];

    for (chunk_idx, ops) in chunk_operations.iter().enumerate().take(chunk_count) {
        for &op in ops {
            for &pred in &predecessors[op] {
                let pred_chunk = op_to_chunk[pred];
                if pred_chunk != chunk_idx {
                    chunk_predecessors[chunk_idx].push(pred_chunk);
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
        for (idx, chunk) in chunks.iter_mut().enumerate().take(chunk_count) {
            chunk.is_output = true;
            output_chunks.push(idx);
        }
    }

    if root_chunks.is_empty() {
        return Err(VKMLError::GraphScheduler(
            "Execution plan contains no root chunks".into(),
        ));
    }

    // snapshot devices for dependents checks to avoid mutable/immutable borrow conflicts
    // TODO: Clean this up
    let devices_snapshot: Vec<DeviceId> = chunks.iter().map(|c| c.device.clone()).collect();

    for chunk in chunks.iter_mut().take(chunk_count) {
        let needs_wait = match chunk.device {
            DeviceId::Gpu(gpu_idx) => {
                if chunk.is_output {
                    true
                } else {
                    chunk
                        .dependents
                        .iter()
                        .any(|&dep| match devices_snapshot[dep] {
                            DeviceId::Gpu(dep_gpu) => dep_gpu != gpu_idx,
                            DeviceId::Cpu => true,
                        })
                }
            }
            DeviceId::Cpu => false,
        };
        chunk.needs_host_wait = needs_wait;
    }

    Ok(ExecutionPlan {
        chunks,
        output_chunks,
        root_chunks,
    })
}
