use std::collections::VecDeque;
use std::sync::{
    Arc, Condvar, Mutex, OnceLock, Weak,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

use vulkanalia::vk;
use zero_pool::{global_pool, zp_define_task_fn};

use crate::compute::compute_manager::ComputeManager;
use crate::tensor::tensor::DeviceId;
use crate::tensor_graph::tensor_graph::{OperationId, TensorGraph};
use crate::utils::error::VKMLError;
use vulkanalia::vk::DeviceV1_0;

pub type ChunkId = usize;

pub struct DynamicExecutionChunk {
    pub device: DeviceId,
    pub operations: Vec<OperationId>,
    pub predecessors: Vec<ChunkId>,
    pub dependents: Vec<ChunkId>,
    pub initial_dep_count: usize,
    pub is_output: bool,
    pub needs_host_wait: bool,
}

pub struct DynamicExecutionPlan {
    pub chunks: Vec<DynamicExecutionChunk>,
    pub operation_to_chunk: Vec<ChunkId>,
    pub output_chunks: Vec<ChunkId>,
    pub root_chunks: Vec<ChunkId>,
    cached_chunk_command_buffers: Vec<OnceLock<Vec<vk::CommandBuffer>>>,
}

impl DynamicExecutionPlan {
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }
}

pub fn create_dynamic_execution_plan(
    compute_manager: &ComputeManager,
) -> Result<DynamicExecutionPlan, VKMLError> {
    let tensor_graph = &compute_manager.tensor_graph;
    if tensor_graph.operations.is_empty() {
        return Err(VKMLError::Generic(
            "Dynamic scheduler cannot execute an empty graph".into(),
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
            if let Some(producer) = tensor_producers[tensor_id] {
                if producer != op_idx && seen_markers[producer] != current_mark {
                    seen_markers[producer] = current_mark;
                    predecessors[op_idx].push(producer);
                    successors[producer].push(op_idx);
                }
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
            "Cycle detected while building dynamic execution plan".into(),
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

    let mut chunks: Vec<DynamicExecutionChunk> = Vec::new();
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

        chunks.push(DynamicExecutionChunk {
            device,
            operations: chain,
            predecessors: Vec::new(),
            dependents: Vec::new(),
            initial_dep_count: 0,
            is_output: false,
            needs_host_wait: false,
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
            "Dynamic execution plan contains no root chunks".into(),
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

    let chunk_count = chunks.len();
    let cached_chunk_command_buffers: Vec<OnceLock<Vec<vk::CommandBuffer>>> =
        (0..chunk_count).map(|_| OnceLock::new()).collect();

    Ok(DynamicExecutionPlan {
        chunks,
        operation_to_chunk,
        output_chunks,
        root_chunks,
        cached_chunk_command_buffers,
    })
}

struct ExecutionState {
    plan: Arc<DynamicExecutionPlan>,
    compute_manager: *const ComputeManager,
    device_semaphore_offsets: Vec<u64>,
    device_chunk_counters: Vec<AtomicU64>,
    chunk_dependencies_remaining: Vec<AtomicUsize>,
    outputs_remaining: AtomicUsize,
    completion_signal: Arc<(Mutex<()>, Condvar)>,
    chunk_task_params: Vec<ChunkTaskParams>,
}

impl ExecutionState {
    fn new(
        plan: Arc<DynamicExecutionPlan>,
        manager: &ComputeManager,
    ) -> Result<Arc<Self>, VKMLError> {
        let gpu_count = manager.gpu_count();

        let mut device_counts = vec![0u64; gpu_count];
        for chunk in &plan.chunks {
            if let DeviceId::Gpu(idx) = chunk.device {
                device_counts[idx] += 1;
            }
        }

        let mut device_semaphore_offsets = vec![0u64; gpu_count];
        for gpu_idx in 0..gpu_count {
            if device_counts[gpu_idx] > 0 {
                device_semaphore_offsets[gpu_idx] = manager
                    .gpu_ref(gpu_idx)
                    .allocate_semaphore_values(device_counts[gpu_idx]);
            }
        }

        let device_chunk_counters: Vec<AtomicU64> =
            (0..gpu_count).map(|_| AtomicU64::new(0)).collect();

        let chunk_dependencies_remaining: Vec<AtomicUsize> = plan
            .chunks
            .iter()
            .map(|chunk| AtomicUsize::new(chunk.initial_dep_count))
            .collect();

        // Prime the plan-level cache so command buffers are recorded only once per chunk.
        for (chunk_id, chunk) in plan.chunks.iter().enumerate() {
            if plan.cached_chunk_command_buffers[chunk_id].get().is_some() {
                continue;
            }

            let buffers = match chunk.device {
                DeviceId::Gpu(gpu_idx) => {
                    let mut buffers = Vec::with_capacity(chunk.operations.len());
                    for &op_id in &chunk.operations {
                        let buffer = create_gpu_command_buffer(manager, op_id, gpu_idx)?;
                        buffers.push(buffer);
                    }
                    buffers
                }
                DeviceId::Cpu => Vec::new(),
            };

            let _ = plan.cached_chunk_command_buffers[chunk_id].set(buffers);
        }

        let outputs_remaining_init = plan.output_chunks.len();

        let completion_signal = Arc::new((Mutex::new(()), Condvar::new()));

        let plan_for_state = Arc::clone(&plan);
        let manager_ptr = manager as *const ComputeManager;

        let state = Arc::new_cyclic(move |weak_self| {
            let chunk_task_params: Vec<ChunkTaskParams> = (0..plan_for_state.total_chunks())
                .map(|chunk_id| ChunkTaskParams {
                    chunk_id,
                    state: weak_self.clone(),
                })
                .collect();

            ExecutionState {
                plan: Arc::clone(&plan_for_state),
                compute_manager: manager_ptr,
                device_semaphore_offsets,
                device_chunk_counters,
                chunk_dependencies_remaining,
                outputs_remaining: AtomicUsize::new(outputs_remaining_init),
                completion_signal,
                chunk_task_params,
            }
        });

        Ok(state)
    }

    fn submit_initial_chunks(&self) {
        for &chunk_idx in &self.plan.root_chunks {
            self.submit_chunk(chunk_idx);
        }
    }

    fn submit_chunk(&self, chunk_id: ChunkId) {
        let params = &self.chunk_task_params[chunk_id];
        global_pool().submit_task(chunk_execute_task, params);
    }

    fn execute_chunk(&self, chunk_id: ChunkId) -> Result<(), VKMLError> {
        let compute_manager = unsafe { &*self.compute_manager };
        let chunk = &self.plan.chunks[chunk_id];

        match &chunk.device {
            DeviceId::Gpu(gpu_idx) => {
                self.execute_gpu_chunk(chunk_id, *gpu_idx, compute_manager)?;
            }
            DeviceId::Cpu => {
                self.execute_cpu_chunk(chunk_id, compute_manager)?;
                self.finalize_chunk(chunk_id);
            }
        }

        Ok(())
    }

    fn execute_gpu_chunk(
        &self,
        chunk_id: ChunkId,
        gpu_idx: usize,
        compute_manager: &ComputeManager,
    ) -> Result<(), VKMLError> {
        let gpu = compute_manager.gpu_ref(gpu_idx);

        let local_index = self.device_chunk_counters[gpu_idx].fetch_add(1, Ordering::Relaxed);
        let signal_value = self.device_semaphore_offsets[gpu_idx] + local_index;

        let command_buffers = self.plan.cached_chunk_command_buffers[chunk_id]
            .get()
            .ok_or_else(|| {
                VKMLError::Generic(format!(
                    "Missing command buffers for chunk {} on GPU {}",
                    chunk_id, gpu_idx
                ))
            })?
            .as_slice();
        let wait_slice: &[(vk::Semaphore, u64)] = &[];
        gpu.submit_with_timeline_semaphore(command_buffers, wait_slice, signal_value)?;

        if self.plan.chunks[chunk_id].needs_host_wait {
            // Block this worker until the GPU signals completion so dependents see consistent state.
            if let Err(err) = gpu.wait_for_timeline_value(signal_value) {
                return Err(err);
            }
        }

        self.finalize_chunk(chunk_id);
        Ok(())
    }

    fn execute_cpu_chunk(
        &self,
        chunk_id: ChunkId,
        compute_manager: &ComputeManager,
    ) -> Result<(), VKMLError> {
        let chunk = &self.plan.chunks[chunk_id];
        for &op_id in &chunk.operations {
            let instruction = compute_manager.tensor_graph.get_instruction_or_panic(op_id);
            instruction.execute_cpu(compute_manager);
        }
        Ok(())
    }

    fn mark_output_complete(&self) {
        if self.outputs_remaining.fetch_sub(1, Ordering::Release) == 1 {
            self.signal_completion();
        }
    }

    fn signal_completion(&self) {
        let (lock, cvar) = &*self.completion_signal;
        let guard = lock.lock().unwrap();
        drop(guard);
        cvar.notify_one();
    }

    fn await_completion(&self) {
        let (lock, cvar) = &*self.completion_signal;
        let mut guard = lock.lock().unwrap();
        while self.outputs_remaining.load(Ordering::Acquire) != 0 {
            guard = cvar.wait(guard).unwrap();
        }
    }

    fn finalize_chunk(&self, chunk_id: ChunkId) {
        let chunk = &self.plan.chunks[chunk_id];

        if chunk.is_output {
            self.mark_output_complete();
        }

        for &dependent in &chunk.dependents {
            let previous =
                self.chunk_dependencies_remaining[dependent].fetch_sub(1, Ordering::Release);
            if previous == 1 {
                self.submit_chunk(dependent);
            }
        }
    }
}

struct ChunkTaskParams {
    chunk_id: ChunkId,
    state: Weak<ExecutionState>,
}

zp_define_task_fn!(chunk_execute_task, ChunkTaskParams, |params| {
    let Some(state) = params.state.upgrade() else {
        return;
    };
    let chunk_id = params.chunk_id;

    if let Err(err) = state.execute_chunk(chunk_id) {
        state.signal_completion();
        panic!("execute_chunk failed: {err}");
    }
});

pub fn execute_dynamic_plan(
    compute_manager: &ComputeManager,
    plan: Arc<DynamicExecutionPlan>,
) -> Result<(), VKMLError> {
    let state = ExecutionState::new(plan, compute_manager)?;

    state.submit_initial_chunks();
    state.await_completion();

    Ok(())
}

pub fn plan_requires_rebuild(plan: &DynamicExecutionPlan, graph: &TensorGraph) -> bool {
    plan.operation_to_chunk.len() != graph.operations.len()
}

fn create_gpu_command_buffer(
    compute_manager: &ComputeManager,
    op_id: OperationId,
    gpu_idx: usize,
) -> Result<vk::CommandBuffer, VKMLError> {
    let gpu = compute_manager.gpu_ref(gpu_idx);

    unsafe {
        let alloc_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            next: std::ptr::null(),
            command_pool: gpu.get_command_pool(),
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
        };

        let buffers = gpu
            .get_device()
            .allocate_command_buffers(&alloc_info)
            .map_err(|err| {
                VKMLError::VulkanError(format!(
                    "Failed to allocate command buffer for op {} on GPU {}: {}",
                    op_id, gpu_idx, err
                ))
            })?;

        let command_buffer = buffers.into_iter().next().ok_or_else(|| {
            VKMLError::VulkanError(format!(
                "No command buffer returned for op {} on GPU {}",
                op_id, gpu_idx
            ))
        })?;

        let instruction = compute_manager.tensor_graph.get_instruction_or_panic(op_id);

        gpu.begin_command_buffer(command_buffer).map_err(|err| {
            VKMLError::VulkanError(format!(
                "Failed to begin command buffer for op {}: {}",
                op_id, err
            ))
        })?;

        instruction
            .record_into_command_buffer(gpu, command_buffer, compute_manager)
            .map_err(|err| {
                VKMLError::VulkanError(format!(
                    "Failed to record commands for op {}: {}",
                    op_id, err
                ))
            })?;

        gpu.end_command_buffer(command_buffer).map_err(|err| {
            VKMLError::VulkanError(format!(
                "Failed to end command buffer for op {}: {}",
                op_id, err
            ))
        })?;

        Ok(command_buffer)
    }
}
