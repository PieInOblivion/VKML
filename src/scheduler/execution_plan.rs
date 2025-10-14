use std::collections::VecDeque;
use std::sync::OnceLock;

use crate::compute::compute_manager::ComputeManager;
use crate::tensor::tensor::DeviceId;
use crate::tensor_graph::tensor_graph::OperationId;
use crate::utils::error::VKMLError;
use vulkanalia::vk;

pub type ChunkId = usize;

pub struct ExecutionChunk {
    pub device: DeviceId,
    pub operations: Vec<OperationId>,
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

pub fn create_execution_plan(compute_manager: &ComputeManager) -> Result<ExecutionPlan, VKMLError> {
    let tensor_graph = &compute_manager.tensor_graph;
    if tensor_graph.operations.is_empty() {
        return Err(VKMLError::Generic(
            "Scheduler cannot execute an empty graph".into(),
        ));
    }

    let op_count = tensor_graph.operations.len();
    let tensor_count = tensor_graph.tensor_descs.len();
    let mut tensor_producers: Vec<Option<OperationId>> = vec![None; tensor_count];
    for (op_idx, op) in tensor_graph.operations.iter().enumerate() {
        for &tid in &op.get_output_tensor_ids() {
            debug_assert!(tid < tensor_count, "tensor id out of range");
            tensor_producers[tid] = Some(op_idx);
        }
    }

    let mut predecessors: Vec<Vec<OperationId>> = vec![Vec::new(); op_count];
    let mut successors: Vec<Vec<OperationId>> = vec![Vec::new(); op_count];

    let mut seen_markers = vec![0u32; op_count];
    let mut current_mark: u32 = 1;

    for op_idx in 0..op_count {
        let op = &tensor_graph.operations[op_idx];
        current_mark = current_mark.wrapping_add(1);
        if current_mark == 0 {
            seen_markers.fill(0);
            current_mark = 1;
        }
        for &tensor_id in &op.get_input_tensor_ids() {
            if tensor_id >= tensor_count {
                continue;
            }
            if let Some(producer) = tensor_producers[tensor_id]
                && producer != op_idx
                && seen_markers[producer] != current_mark
            {
                seen_markers[producer] = current_mark;
                predecessors[op_idx].push(producer);
                successors[producer].push(op_idx);
            }
        }
    }

    let mut in_degrees: Vec<usize> = predecessors.iter().map(|p| p.len()).collect();
    let mut queue: VecDeque<OperationId> =
        (0..op_count).filter(|&op| in_degrees[op] == 0).collect();
    let mut topo_order = Vec::with_capacity(op_count);

    while let Some(op) = queue.pop_front() {
        topo_order.push(op);
        for &succ in &successors[op] {
            if in_degrees[succ] == 0 {
                continue;
            }
            in_degrees[succ] -= 1;
            if in_degrees[succ] == 0 {
                queue.push_back(succ);
            }
        }
    }

    if topo_order.len() != op_count {
        return Err(VKMLError::Generic(
            "Cycle detected while building execution plan".into(),
        ));
    }

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

    for &op in &topo_order {
        if operation_to_chunk[op] != usize::MAX {
            continue;
        }

        let device = op_devices[op].clone();
        let mut chain: Vec<OperationId> = Vec::new();
        let mut current = op;

        chain_mark_value = chain_mark_value.wrapping_add(1);
        if chain_mark_value == 0 {
            chain_marks.fill(0);
            chain_mark_value = 1;
        }

        loop {
            chain_marks[current] = chain_mark_value;
            chain.push(current);

            let mut candidate: Option<OperationId> = None;
            for &succ in &successors[current] {
                if operation_to_chunk[succ] != usize::MAX {
                    continue;
                }
                if chain_marks[succ] == chain_mark_value {
                    continue;
                }
                if op_devices[succ] != device {
                    continue;
                }
                if !predecessors[succ].iter().all(|&pred| {
                    operation_to_chunk[pred] != usize::MAX || chain_marks[pred] == chain_mark_value
                }) {
                    continue;
                }
                if candidate.is_some() {
                    candidate = None;
                    break;
                }
                candidate = Some(succ);
            }

            match candidate {
                Some(next) => {
                    current = next;
                }
                None => break,
            }
        }

        let chunk_id = chunks.len();
        for &op_id in &chain {
            operation_to_chunk[op_id] = chunk_id;
        }

        chunks.push(ExecutionChunk {
            device,
            operations: chain,
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
    let mut root_chunks: Vec<ChunkId> = Vec::new();

    for chunk_idx in 0..chunk_count {
        for &op in &chunks[chunk_idx].operations {
            for &pred_op in &predecessors[op] {
                let pred_chunk = operation_to_chunk[pred_op];
                if pred_chunk != chunk_idx {
                    chunk_predecessors[chunk_idx].push(pred_chunk);
                }
            }
        }
    }

    for chunk_idx in 0..chunk_count {
        let preds = &mut chunk_predecessors[chunk_idx];
        preds.sort_unstable();
        preds.dedup();
    }

    let mut chunk_dependents: Vec<Vec<ChunkId>> = vec![Vec::new(); chunk_count];
    for (chunk_idx, preds) in chunk_predecessors.iter().enumerate() {
        for &pred in preds {
            chunk_dependents[pred].push(chunk_idx);
        }
    }

    for dependents in &mut chunk_dependents {
        dependents.sort_unstable();
        dependents.dedup();
    }

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
        for &op_id in &chunk.operations {
            let op = &tensor_graph.operations[op_id];
            if op
                .get_output_tensor_ids()
                .iter()
                .any(|&tid| tid < output_tensor_flags.len() && output_tensor_flags[tid])
            {
                is_output = true;
                break;
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
