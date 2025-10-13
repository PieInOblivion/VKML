use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use vulkanalia::vk;
use zero_pool::{global_pool, zp_define_task_fn};

use crate::compute::compute_manager::ComputeManager;
use crate::tensor::tensor::DeviceId;
use crate::tensor_graph::tensor_graph::{OperationId, TensorGraph};
use crate::utils::error::VKMLError;

pub type ChunkId = usize;

#[derive(Debug, Clone)]
pub struct DynamicExecutionChunk {
    pub device: DeviceId,
    pub operations: Vec<OperationId>,
    #[allow(dead_code)]
    pub internal_dependencies: Vec<(usize, usize)>,
    pub predecessors: Vec<ChunkId>,
    pub dependents: Vec<ChunkId>,
    pub initial_dep_count: usize,
    pub is_output: bool,
}

#[derive(Debug, Clone)]
pub struct DynamicExecutionPlan {
    pub chunks: Vec<DynamicExecutionChunk>,
    pub operation_to_chunk: Vec<ChunkId>,
    pub output_chunks: Vec<ChunkId>,
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
    let mut tensor_producers: HashMap<usize, OperationId> = HashMap::new();
    for (op_idx, op) in tensor_graph.operations.iter().enumerate() {
        for &tid in &op.get_output_tensor_ids() {
            tensor_producers.insert(tid, op_idx);
        }
    }

    let mut predecessors: Vec<Vec<OperationId>> = vec![Vec::new(); op_count];
    let mut successors: Vec<Vec<OperationId>> = vec![Vec::new(); op_count];

    for op_idx in 0..op_count {
        let op = &tensor_graph.operations[op_idx];
        let mut seen = HashSet::new();
        for &tensor_id in &op.get_input_tensor_ids() {
            if let Some(&producer) = tensor_producers.get(&tensor_id) {
                if producer != op_idx && seen.insert(producer) {
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

    for &op in &topo_order {
        if operation_to_chunk[op] != usize::MAX {
            continue;
        }

        let device = op_devices[op].clone();
        let mut chain: Vec<OperationId> = Vec::new();
        let mut chain_set: HashSet<OperationId> = HashSet::new();
        let mut current = op;

        loop {
            chain_set.insert(current);
            chain.push(current);

            let mut candidate: Option<OperationId> = None;
            for &succ in &successors[current] {
                if operation_to_chunk[succ] != usize::MAX || chain_set.contains(&succ) {
                    continue;
                }
                if op_devices[succ] != device {
                    continue;
                }
                if !predecessors[succ].iter().all(|&pred| {
                    operation_to_chunk[pred] != usize::MAX || chain_set.contains(&pred)
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

        let mut index_map: HashMap<OperationId, usize> = HashMap::new();
        for (idx, &op_id) in chain.iter().enumerate() {
            index_map.insert(op_id, idx);
        }

        let mut internal_deps: Vec<(usize, usize)> = Vec::new();
        for &src in &chain {
            let src_idx = index_map[&src];
            for &dst in &successors[src] {
                if let Some(&dst_idx) = index_map.get(&dst) {
                    internal_deps.push((src_idx, dst_idx));
                }
            }
        }

        let chunk_id = chunks.len();
        for &op_id in &chain {
            operation_to_chunk[op_id] = chunk_id;
        }

        chunks.push(DynamicExecutionChunk {
            device,
            operations: chain,
            internal_dependencies: internal_deps,
            predecessors: Vec::new(),
            dependents: Vec::new(),
            initial_dep_count: 0,
            is_output: false,
        });
    }

    let chunk_count = chunks.len();
    let mut chunk_predecessors: Vec<HashSet<ChunkId>> = vec![HashSet::new(); chunk_count];

    for chunk_idx in 0..chunk_count {
        for &op in &chunks[chunk_idx].operations {
            for &pred_op in &predecessors[op] {
                let pred_chunk = operation_to_chunk[pred_op];
                if pred_chunk != chunk_idx {
                    chunk_predecessors[chunk_idx].insert(pred_chunk);
                }
            }
        }
    }

    let mut chunk_dependents: Vec<HashSet<ChunkId>> = vec![HashSet::new(); chunk_count];
    for (chunk_idx, preds) in chunk_predecessors.iter().enumerate() {
        for &pred in preds {
            chunk_dependents[pred].insert(chunk_idx);
        }
    }

    for chunk_idx in 0..chunk_count {
        let preds: Vec<ChunkId> = {
            let mut vec: Vec<ChunkId> = chunk_predecessors[chunk_idx].iter().copied().collect();
            vec.sort_unstable();
            vec
        };
        let dependents: Vec<ChunkId> = {
            let mut vec: Vec<ChunkId> = chunk_dependents[chunk_idx].iter().copied().collect();
            vec.sort_unstable();
            vec
        };

        chunks[chunk_idx].initial_dep_count = preds.len();
        chunks[chunk_idx].predecessors = preds;
        chunks[chunk_idx].dependents = dependents;
    }

    let output_tensor_ids: HashSet<usize> = tensor_graph
        .get_output_tensor_ids()
        .iter()
        .copied()
        .collect();
    let mut output_chunks: HashSet<ChunkId> = HashSet::new();

    for (chunk_idx, chunk) in chunks.iter_mut().enumerate() {
        let mut is_output = false;
        for &op_id in &chunk.operations {
            let op = &tensor_graph.operations[op_id];
            if op
                .get_output_tensor_ids()
                .iter()
                .any(|tid| output_tensor_ids.contains(tid))
            {
                is_output = true;
                break;
            }
        }
        chunk.is_output = is_output;
        if is_output {
            output_chunks.insert(chunk_idx);
        }
    }

    if output_chunks.is_empty() {
        for idx in 0..chunk_count {
            chunks[idx].is_output = true;
            output_chunks.insert(idx);
        }
    }

    if !chunks.iter().any(|chunk| chunk.initial_dep_count == 0) {
        return Err(VKMLError::Generic(
            "Dynamic execution plan contains no root chunks".into(),
        ));
    }

    Ok(DynamicExecutionPlan {
        chunks,
        operation_to_chunk,
        output_chunks: output_chunks.into_iter().collect(),
    })
}

struct CompletionFlag {
    completed: bool,
}

struct ExecutionState {
    plan: Arc<DynamicExecutionPlan>,
    compute_manager: *const ComputeManager,
    device_semaphore_offsets: Vec<u64>,
    device_chunk_counters: Vec<AtomicU64>,
    chunk_dependencies_remaining: Vec<AtomicUsize>,
    chunk_signal_values: Vec<AtomicU64>,
    chunk_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    outputs_remaining: AtomicUsize,
    completion_signal: Arc<(Mutex<CompletionFlag>, Condvar)>,
    chunk_task_params: Vec<ChunkTaskParams>,
    has_error: AtomicBool,
    error: Mutex<Option<VKMLError>>,
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

        let chunk_signal_values: Vec<AtomicU64> = (0..plan.total_chunks())
            .map(|_| AtomicU64::new(0))
            .collect();

        let mut chunk_command_buffers: Vec<Vec<vk::CommandBuffer>> =
            Vec::with_capacity(plan.total_chunks());
        for chunk in &plan.chunks {
            match chunk.device {
                DeviceId::Gpu(gpu_idx) => {
                    let mut buffers = Vec::with_capacity(chunk.operations.len());
                    for &op_id in &chunk.operations {
                        let buffer = manager.ensure_gpu_command_buffer(op_id, gpu_idx)?;
                        buffers.push(buffer);
                    }
                    chunk_command_buffers.push(buffers);
                }
                DeviceId::Cpu => chunk_command_buffers.push(Vec::new()),
            }
        }

        let outputs_remaining_init = plan.output_chunks.len();

        let completion_signal = Arc::new((
            Mutex::new(CompletionFlag { completed: false }),
            Condvar::new(),
        ));

        let plan_for_state = Arc::clone(&plan);
        let manager_ptr = manager as *const ComputeManager;

        let state = Arc::new_cyclic(move |weak_self| {
            let state_ptr: *const ExecutionState = weak_self.as_ptr();
            debug_assert!(!state_ptr.is_null());

            let chunk_task_params: Vec<ChunkTaskParams> = (0..plan_for_state.total_chunks())
                .map(|chunk_id| ChunkTaskParams {
                    chunk_id,
                    state_ptr,
                })
                .collect();

            ExecutionState {
                plan: Arc::clone(&plan_for_state),
                compute_manager: manager_ptr,
                device_semaphore_offsets,
                device_chunk_counters,
                chunk_dependencies_remaining,
                chunk_signal_values,
                chunk_command_buffers,
                outputs_remaining: AtomicUsize::new(outputs_remaining_init),
                completion_signal,
                chunk_task_params,
                has_error: AtomicBool::new(false),
                error: Mutex::new(None),
            }
        });

        Ok(state)
    }

    fn submit_initial_chunks(&self) {
        for (chunk_idx, chunk) in self.plan.chunks.iter().enumerate() {
            if chunk.initial_dep_count == 0 {
                self.submit_chunk(chunk_idx);
            }
        }
    }

    fn submit_chunk(&self, chunk_id: ChunkId) {
        if self.is_cancelled() {
            return;
        }

        let params = &self.chunk_task_params[chunk_id];
        let future = global_pool().submit_task(chunk_execute_task, params);
        drop(future);
    }

    fn is_cancelled(&self) -> bool {
        self.has_error.load(Ordering::Acquire)
    }

    fn execute_chunk(&self, chunk_id: ChunkId) -> Result<(), VKMLError> {
        if self.is_cancelled() {
            return Ok(());
        }

        let compute_manager = unsafe { &*self.compute_manager };
        let chunk = &self.plan.chunks[chunk_id];

        match &chunk.device {
            DeviceId::Gpu(gpu_idx) => self.execute_gpu_chunk(chunk_id, *gpu_idx, compute_manager),
            DeviceId::Cpu => self.execute_cpu_chunk(chunk_id, compute_manager),
        }
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
        self.chunk_signal_values[chunk_id].store(signal_value, Ordering::Relaxed);

        let command_buffers = &self.chunk_command_buffers[chunk_id];

        gpu.submit_with_timeline_semaphore(command_buffers, &[], signal_value)?;
        gpu.wait_for_timeline_value(signal_value)?;

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

    fn notify_dependents(&self, chunk_id: ChunkId) {
        if self.is_cancelled() {
            return;
        }

        for &dependent in &self.plan.chunks[chunk_id].dependents {
            let previous =
                self.chunk_dependencies_remaining[dependent].fetch_sub(1, Ordering::AcqRel);
            if previous == 1 {
                self.submit_chunk(dependent);
            }
        }
    }

    fn mark_output_complete(&self) {
        if self.outputs_remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.signal_completion();
        }
    }

    fn signal_completion(&self) {
        let (lock, cvar) = &*self.completion_signal;
        let mut guard = lock.lock().unwrap();
        if !guard.completed {
            guard.completed = true;
            cvar.notify_one();
        }
    }

    fn record_error(&self, error: VKMLError) {
        let already_cancelled = self.has_error.swap(true, Ordering::AcqRel);
        if already_cancelled {
            return;
        }

        {
            let mut guard = self.error.lock().unwrap();
            *guard = Some(error);
        }

        self.signal_completion();
    }

    fn await_completion(&self) {
        let (lock, cvar) = &*self.completion_signal;
        let mut guard = lock.lock().unwrap();
        while !guard.completed {
            guard = cvar.wait(guard).unwrap();
        }
    }

    fn take_error(&self) -> Option<VKMLError> {
        let mut guard = self.error.lock().unwrap();
        guard.take()
    }
}

struct ChunkTaskParams {
    chunk_id: ChunkId,
    state_ptr: *const ExecutionState,
}

zp_define_task_fn!(chunk_execute_task, ChunkTaskParams, |params| {
    let state = unsafe { &*params.state_ptr };
    let chunk_id = params.chunk_id;

    let result = state.execute_chunk(chunk_id);

    match result {
        Ok(()) => {
            if !state.is_cancelled() {
                if state.plan.chunks[chunk_id].is_output {
                    state.mark_output_complete();
                }
                state.notify_dependents(chunk_id);
            }
        }
        Err(err) => {
            state.record_error(err);
        }
    }
});

pub fn execute_dynamic_plan(
    compute_manager: &ComputeManager,
    plan: Arc<DynamicExecutionPlan>,
) -> Result<(), VKMLError> {
    let state = ExecutionState::new(plan, compute_manager)?;

    if state.plan.output_chunks.is_empty() {
        state.signal_completion();
        state.await_completion();
        return state.take_error().map_or(Ok(()), Err);
    }

    state.submit_initial_chunks();
    state.await_completion();

    if let Some(error) = state.take_error() {
        return Err(error);
    }

    Ok(())
}

pub fn plan_requires_rebuild(plan: &DynamicExecutionPlan, graph: &TensorGraph) -> bool {
    plan.operation_to_chunk.len() != graph.operations.len()
}
