use std::sync::{
    Arc, Condvar, Mutex, Weak,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

use vulkanalia::vk;
use vulkanalia::vk::DeviceV1_0;
use zero_pool::{global_pool, zp_define_task_fn};

use crate::tensor::tensor::DeviceId;
use crate::tensor_graph::tensor_graph::OperationId;
use crate::utils::error::VKMLError;
use crate::{compute::compute_manager::ComputeManager, scheduler::execution_plan::ChunkId};

use super::execution_plan::ExecutionPlan;

struct ExecutionState {
    plan: Arc<ExecutionPlan>,
    compute_manager: *const ComputeManager,
    device_semaphore_offsets: Vec<u64>,
    device_chunk_counters: Vec<AtomicU64>,
    chunk_dependencies_remaining: Vec<AtomicUsize>,
    outputs_remaining: AtomicUsize,
    completion_signal: Arc<(Mutex<()>, Condvar)>,
    chunk_task_params: Vec<ChunkTaskParams>,
}

impl ExecutionState {
    fn new(plan: Arc<ExecutionPlan>, manager: &ComputeManager) -> Result<Arc<Self>, VKMLError> {
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
        for chunk in &plan.chunks {
            if chunk.command_buffer.get().is_some() {
                continue;
            }

            if let DeviceId::Gpu(gpu_idx) = chunk.device {
                let buffer = create_gpu_chunk_command_buffer(manager, &chunk.operations, gpu_idx)?;

                let _ = chunk.command_buffer.set(buffer);
            };
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

        let command_buffer = self.plan.chunks[chunk_id]
            .command_buffer
            .get()
            .ok_or_else(|| {
                VKMLError::Generic(format!(
                    "Missing command buffers for chunk {} on GPU {}",
                    chunk_id, gpu_idx
                ))
            })?;
        let command_buffers = std::slice::from_ref(command_buffer);
        let wait_slice: &[(vk::Semaphore, u64)] = &[];
        gpu.submit_with_timeline_semaphore(command_buffers, wait_slice, signal_value)?;

        if self.plan.chunks[chunk_id].needs_host_wait {
            // Block this worker until the GPU signals completion so dependents see consistent state.
            gpu.wait_for_timeline_value(signal_value)?
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

pub fn execute_plan(
    compute_manager: &ComputeManager,
    plan: Arc<ExecutionPlan>,
) -> Result<(), VKMLError> {
    let state = ExecutionState::new(plan, compute_manager)?;

    state.submit_initial_chunks();
    state.await_completion();

    Ok(())
}

fn create_gpu_chunk_command_buffer(
    compute_manager: &ComputeManager,
    operations: &[OperationId],
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
                    "Failed to allocate command buffer for chunk on GPU {}: {}",
                    gpu_idx, err
                ))
            })?;

        let command_buffer = buffers.into_iter().next().ok_or_else(|| {
            VKMLError::VulkanError(format!(
                "No command buffer returned for chunk on GPU {}",
                gpu_idx
            ))
        })?;

        let tensor_count = compute_manager.tensor_graph.tensor_descs.len();
        let mut dirty_flags = vec![false; tensor_count];
        let mut dirty_list: Vec<usize> = Vec::new();

        let clear_dirty = |flags: &mut [bool], list: &mut Vec<usize>| {
            for &tid in list.iter() {
                if tid < flags.len() {
                    flags[tid] = false;
                }
            }
            list.clear();
        };

        gpu.begin_command_buffer(command_buffer).map_err(|err| {
            VKMLError::VulkanError(format!(
                "Failed to begin command buffer for GPU {}: {}",
                gpu_idx, err
            ))
        })?;

        for &op_id in operations {
            let instruction = compute_manager.tensor_graph.get_instruction_or_panic(op_id);

            let inputs = instruction.get_input_tensor_ids();
            let needs_barrier = inputs
                .iter()
                .any(|&tid| tid < dirty_flags.len() && dirty_flags[tid]);

            if needs_barrier {
                gpu.barrier_compute_shader_access(command_buffer);
                clear_dirty(&mut dirty_flags, &mut dirty_list);
            }

            instruction
                .record_into_command_buffer(gpu, command_buffer, compute_manager)
                .map_err(|err| {
                    VKMLError::VulkanError(format!(
                        "Failed to record commands for op {}: {}",
                        op_id, err
                    ))
                })?;

            for &tid in instruction.get_output_tensor_ids().iter() {
                if tid < dirty_flags.len() && !dirty_flags[tid] {
                    dirty_flags[tid] = true;
                    dirty_list.push(tid);
                }
            }
        }

        gpu.end_command_buffer(command_buffer).map_err(|err| {
            VKMLError::VulkanError(format!(
                "Failed to end command buffer for GPU {}: {}",
                gpu_idx, err
            ))
        })?;

        Ok(command_buffer)
    }
}
