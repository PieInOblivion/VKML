use std::ptr;
use std::sync::OnceLock;

use crate::compute::{print_model_stats, print_tensorgraph_stats};
use crate::gpu::pool::GpuPool;
use crate::importers::onnx_parser::OnnxParser;
use crate::instruction;
use crate::tensor::cell::TensorCell;
use crate::tensor::tensor::{DeviceId, Tensor};
use crate::utils::error::VKMLError;
use onnx_extractor::OnnxModel;
use zero_pool::{global_pool, zp_define_task_fn};

use crate::instruction::instruction::Instruction;
use crate::tensor_graph::tensor_graph::{ExecutionChunk, ExecutionPlan, OperationId, TensorGraph};
use crate::{
    model::{graph_model::GraphModel, layer_connection::LayerId},
    tensor::desc::TensorDesc,
};
use vulkanalia::vk::{self, CommandBuffer, DeviceV1_0};

use super::cpu_compute::CPUCompute;

pub struct ComputeManager {
    pub tensors: Vec<TensorCell>,

    pub model: GraphModel,
    pub tensor_graph: TensorGraph,

    gpus: GpuPool,
    cpu: CPUCompute,

    // Cached command buffers per operation, indexed by operation ID
    cached_command_buffers: Vec<OnceLock<CommandBuffer>>,
    // Cached execution plan
    cached_execution_plan: Option<ExecutionPlan>,
}

impl ComputeManager {
    pub fn new_from_graph(model: GraphModel) -> Result<Self, VKMLError> {
        Self::new_from_graph_with(model, None, None)
    }

    pub fn new_from_graph_with(
        mut model: GraphModel,
        explicit_gpus: Option<Vec<usize>>,
        cpu_memory_limit_bytes: Option<u64>,
    ) -> Result<Self, VKMLError> {
        if model.verified.is_none() {
            model.verify()?;
        }

        let cpu = CPUCompute::new(cpu_memory_limit_bytes);

        let tensor_graph = TensorGraph::from_graph_model(&model)?;

        let mut manager = Self {
            tensors: Vec::new(),
            model,
            tensor_graph,
            gpus: GpuPool::new(explicit_gpus)?,
            cpu,
            cached_command_buffers: Vec::new(),
            cached_execution_plan: None,
        };

        let total_memory = manager.tensor_graph.memory_requirements as u64;
        let total_available: u64 = manager
            .gpus
            .gpus()
            .iter()
            .map(|gpu| gpu.available_memory())
            .sum::<u64>()
            + manager.cpu.memory_tracking.get_available();

        if total_memory > total_available {
            return Err(VKMLError::Generic(format!(
                "Model requires {} bytes but only {} available",
                total_memory, total_available
            )));
        }

        manager.allocate_tensor_graph(Vec::new())?;

        Ok(manager)
    }

    pub fn new_from_onnx_path(onnx_path: &str) -> Result<Self, VKMLError> {
        Self::new_from_onnx_path_with(onnx_path, None, None, 1)
    }

    /// Create ComputeManager from ONNX file with custom settings
    pub fn new_from_onnx_path_with(
        onnx_path: &str,
        explicit_gpus: Option<Vec<usize>>,
        cpu_memory_limit_bytes: Option<u64>,
        batch_size: usize,
    ) -> Result<Self, VKMLError> {
        assert!(batch_size > 0, "batch_size must be greater than 0");

        let onnx_model = OnnxModel::load_from_file(onnx_path).map_err(|e| {
            VKMLError::OnnxImporterError(format!(
                "Failed to load ONNX model from '{}': {}",
                onnx_path, e
            ))
        })?;

        let (tensor_graph, tensor_bytes) =
            OnnxParser::parse_onnx_model(onnx_model, batch_size as i64)?;

        Self::new_from_tensor_graph(
            tensor_graph,
            tensor_bytes,
            GpuPool::new(explicit_gpus)?,
            cpu_memory_limit_bytes,
        )
    }

    fn new_from_tensor_graph(
        tensor_graph: TensorGraph,
        tensor_bytes: Vec<Option<Box<[u8]>>>,
        gpus: GpuPool,
        cpu_memory_limit_bytes: Option<u64>,
    ) -> Result<Self, VKMLError> {
        let cpu = CPUCompute::new(cpu_memory_limit_bytes);

        // TODO: Implement one type of model representation.
        // Placeholder minimal model until graph-only mode is supported
        let model = GraphModel::new(1);

        let mut manager = Self {
            gpus,
            cpu,
            tensors: Vec::new(),
            model,
            tensor_graph,
            cached_command_buffers: Vec::new(),
            cached_execution_plan: None,
        };

        let total_memory = manager.tensor_graph.memory_requirements as u64;
        let total_available: u64 = manager
            .gpus
            .gpus()
            .iter()
            .map(|gpu| gpu.available_memory())
            .sum::<u64>()
            + manager.cpu.memory_tracking.get_available();

        if total_memory > total_available {
            return Err(VKMLError::Generic(format!(
                "Model requires {} bytes but only {} available",
                total_memory, total_available
            )));
        }

        manager.allocate_tensor_graph(tensor_bytes)?;
        Ok(manager)
    }

    // This is essentially a graph partitioning problem.
    // This current approach is a greedy approach that may not fit best for most models,
    // but it is quick to compute, and good enough in most cases.
    // An example of where it doesn't work is for example feeding the algorithm two isolated graphs,
    // while allocating them fully on seperate GPUs would be best, this will allocate half each
    // on each gpu.
    // Requirements for this function as currently designed.
    // There's mostly two stages of optimisation for the flattened tensor graph
    // The execution plan which plans parrallel compute
    // and this tensor allocation stratagy.
    // They might become more intertwined in the future, but currently
    // any planned optimisations can be designed seperately between the two.
    //
    // 1. Allocate tensors in execution order
    // 2. All tensors required for an instruction are on the same device
    // 3. Continue until device is full: The algorithm assigns operations to the current device until it encounters one that won't fit (based on memory tensor memory tracking)
    // 4. When full, allocate transfers on the next device: When a device fills up, the algorithm: Identifies all tensors from the current device that will be needed by future operations. Creates storage tensors for these on the next device. Allocates memory for these transfers before moving any regular operations to the next device
    // 5. Modify the graph as required: The algorithm creates explicit transfer operations in the graph and updates all future operations to use the transferred tensor versions instead of the originals.
    // 6. Continue on next device: After handling transfers, the algorithm moves to the next device and continues the same process - allocating all tensors for each instruction on that device.
    //
    // InputBuffers are not treated any differently, as there are possibilities of there being one in the middle
    // of a graph, and that resulting in it's best placement being not the first device.
    //
    // Future ideas:
    //      - Graph models that have split paths of multiple layers would likely benefit from being executed on seperate gpus?
    //      - Graphs with very large layers might benefit from backpropogation being split between devices?
    //      - 'initialisers' would likely be an enum of Vec<Option<InitType>>, where init type is an instruction or Box<[u8]>
    fn allocate_tensor_graph(
        &mut self,
        initialisers: Vec<Option<Box<[u8]>>>,
    ) -> Result<(), VKMLError> {
        // Get stage-based plan and flatten to a linear sequence of operations
        let execution_plan = self.tensor_graph.create_stage_plan();
        let flattened_ops: Vec<OperationId> = execution_plan.into_iter().flatten().collect();

        // Track planned tensor locations: tensor_id -> DeviceLocation
        let mut tensor_locations: Vec<Option<DeviceId>> =
            vec![None; self.tensor_graph.tensor_descs.len()];

        // Maintain a list of tensor remappings per tensor: tensor_id -> [(device, new_id)]
        // Most tensors have 0-1 remappings, so a small Vec is more efficient than HashMap
        let mut tensor_remappings: Vec<Vec<(DeviceId, usize)>> =
            vec![Vec::new(); self.tensor_graph.tensor_descs.len()];

        // Store remappings needed for operations: indexed by op_id
        let mut operation_remappings: Vec<Option<(Vec<usize>, Vec<usize>)>> =
            vec![None; self.tensor_graph.operations.len()];

        // New tensors created for transfers or device-local outputs - including layer info
        let mut new_tensors: Vec<(TensorDesc, DeviceId, Option<LayerId>)> = Vec::new();

        // Transfer operations to insert: (insert_before_op, transfer_instr)
        let mut transfer_operations: Vec<(OperationId, Box<dyn Instruction>)> = Vec::new();

        // Track available memory per device in the desired order (GPUs then CPU)
        let mut available_memory: Vec<(DeviceId, u64)> = Vec::new();
        for (idx, gpu) in self.gpus.gpus().iter().enumerate() {
            available_memory.push((DeviceId::GPU(idx), gpu.available_memory()));
        }
        available_memory.push((DeviceId::CPU, self.cpu.memory_tracking.get_available()));

        // Helper to get tensor size
        let tensor_size =
            |tid: usize| -> u64 { self.tensor_graph.tensor_descs[tid].size_in_bytes() as u64 };

        // For each operation in execution order, pick the first device (GPUs in order, CPU last)
        for &op_id in &flattened_ops {
            let instruction = &self.tensor_graph.operations[op_id];
            let input_tensors = instruction.get_input_tensor_ids();
            let output_tensors = instruction.get_output_tensor_ids();

            // Compute required memory for this op on a candidate device.
            // We must include sizes for: unallocated tensors and remapped tensors (transfers)
            let mut chosen_device_idx: Option<usize> = None;

            'device_search: for dev_idx in 0..available_memory.len() {
                let cand_device = &available_memory[dev_idx].0;
                let mut needed: u64 = 0;

                // Inputs
                for &tid in &input_tensors {
                    match &tensor_locations[tid] {
                        None => {
                            // unallocated -> will be allocated here
                            needed = needed.saturating_add(tensor_size(tid));
                        }
                        Some(loc) if loc != cand_device => {
                            // already allocated elsewhere -> we need to create a remapped tensor here
                            if !tensor_remappings[tid]
                                .iter()
                                .any(|(dev, _)| dev == cand_device)
                            {
                                needed = needed.saturating_add(tensor_size(tid));
                            }
                        }
                        _ => {}
                    }
                }

                // Outputs
                for &tid in &output_tensors {
                    match &tensor_locations[tid] {
                        None => {
                            needed = needed.saturating_add(tensor_size(tid));
                        }
                        Some(loc) if loc != cand_device => {
                            if !tensor_remappings[tid]
                                .iter()
                                .any(|(dev, _)| dev == cand_device)
                            {
                                needed = needed.saturating_add(tensor_size(tid));
                            }
                        }
                        _ => {}
                    }
                }

                // Check if candidate device has enough available memory
                if needed <= available_memory[dev_idx].1 {
                    chosen_device_idx = Some(dev_idx);
                    break 'device_search;
                }
            }

            let dev_idx = match chosen_device_idx {
                Some(i) => i,
                None => {
                    return Err(VKMLError::Generic(format!(
                        "Operation {:?} cannot fit on any device during planning",
                        op_id
                    )));
                }
            };

            let current_device = available_memory[dev_idx].0.clone();

            // Prepare new input/output lists for remapping
            let mut new_inputs = Vec::with_capacity(input_tensors.len());
            let mut new_outputs = Vec::with_capacity(output_tensors.len());
            let mut remapping_needed = false;

            // Handle inputs
            for &tid in &input_tensors {
                match &tensor_locations[tid] {
                    None => {
                        // allocate original tensor on this device
                        tensor_locations[tid] = Some(current_device.clone());
                        let sz = tensor_size(tid);
                        available_memory[dev_idx].1 =
                            available_memory[dev_idx].1.saturating_sub(sz);
                        new_inputs.push(tid);
                    }
                    Some(loc) if loc != &current_device => {
                        if let Some(&(_, mapped_id)) = tensor_remappings[tid]
                            .iter()
                            .find(|(dev, _)| dev == &current_device)
                        {
                            new_inputs.push(mapped_id);
                            remapping_needed = true;
                        } else {
                            // create a new tensor descriptor for the transfer target
                            let new_tensor_id =
                                self.tensor_graph.tensor_descs.len() + new_tensors.len();
                            let original_desc = &self.tensor_graph.tensor_descs[tid];
                            let original_layer_id = self.tensor_graph.tensor_to_layer[tid];
                            let tensor_desc = original_desc.clone();
                            let sz = tensor_desc.size_in_bytes() as u64;

                            // reserve memory on the chosen device
                            available_memory[dev_idx].1 =
                                available_memory[dev_idx].1.saturating_sub(sz);

                            new_tensors.push((
                                tensor_desc,
                                current_device.clone(),
                                original_layer_id,
                            ));

                            // create transfer instruction from src -> dst and schedule it before this op
                            let src_device = tensor_locations[tid].clone().unwrap();
                            let transfer_instr = instruction::transfer(
                                tid,
                                new_tensor_id,
                                src_device,
                                current_device.clone(),
                            );
                            transfer_operations.push((op_id, transfer_instr));

                            tensor_remappings[tid].push((current_device.clone(), new_tensor_id));
                            new_inputs.push(new_tensor_id);
                            remapping_needed = true;
                        }
                    }
                    _ => {
                        // already on this device
                        new_inputs.push(tid);
                    }
                }
            }

            // Handle outputs
            for &tid in &output_tensors {
                match &tensor_locations[tid] {
                    None => {
                        tensor_locations[tid] = Some(current_device.clone());
                        let sz = tensor_size(tid);
                        available_memory[dev_idx].1 =
                            available_memory[dev_idx].1.saturating_sub(sz);
                        new_outputs.push(tid);
                    }
                    Some(loc) if loc != &current_device => {
                        if let Some(&(_, mapped_id)) = tensor_remappings[tid]
                            .iter()
                            .find(|(dev, _)| dev == &current_device)
                        {
                            new_outputs.push(mapped_id);
                            remapping_needed = true;
                        } else {
                            let new_tensor_id =
                                self.tensor_graph.tensor_descs.len() + new_tensors.len();
                            let original_desc = &self.tensor_graph.tensor_descs[tid];
                            let original_layer_id = self.tensor_graph.tensor_to_layer[tid];
                            let tensor_desc = original_desc.clone();
                            let sz = tensor_desc.size_in_bytes() as u64;

                            available_memory[dev_idx].1 =
                                available_memory[dev_idx].1.saturating_sub(sz);

                            new_tensors.push((
                                tensor_desc,
                                current_device.clone(),
                                original_layer_id,
                            ));
                            tensor_remappings[tid].push((current_device.clone(), new_tensor_id));
                            new_outputs.push(new_tensor_id);
                            remapping_needed = true;
                        }
                    }
                    _ => {
                        new_outputs.push(tid);
                    }
                }
            }

            if remapping_needed {
                operation_remappings[op_id] = Some((new_inputs, new_outputs));
            }
        }

        // 1. Create all new tensor descriptors for transfers (allocation happens later)
        for (tensor_desc, device_location, layer_id) in new_tensors {
            self.tensor_graph.tensor_descs.push(tensor_desc);
            self.tensor_graph.tensor_to_layer.push(layer_id);
            tensor_locations.push(Some(device_location));
        }

        // Ensure initialisers matches the new tensor count by extending with None for newly added tensors
        let mut initialisers = initialisers;
        if initialisers.len() < self.tensor_graph.tensor_descs.len() {
            initialisers.resize(self.tensor_graph.tensor_descs.len(), None);
        }

        // 2. Insert transfer operations into the instruction list.
        // Sort transfers by their original insert position so we can insert in order and compute
        // how many transfers were inserted before any given op.
        transfer_operations.sort_by_key(|(op_idx, _)| *op_idx);

        // Keep a simple vector of the insert positions for fast counting later
        let transfer_positions: Vec<OperationId> =
            transfer_operations.iter().map(|(op, _)| *op).collect();

        let mut inserted = 0usize;
        for (insert_before_op, transfer_instr) in transfer_operations.drain(..) {
            let adjusted_pos = insert_before_op + inserted;
            let layer_id = self.tensor_graph.operation_to_layer[insert_before_op];
            self.tensor_graph
                .operations
                .insert(adjusted_pos, transfer_instr);
            self.tensor_graph
                .operation_to_layer
                .insert(adjusted_pos, layer_id);
            inserted += 1;
        }

        // 3. Apply the tensor remappings to all operations. We need to account for how many
        // transfer ops were inserted before each original op index.
        let remap_entries: Vec<(OperationId, Vec<usize>, Vec<usize>)> = operation_remappings
            .into_iter()
            .enumerate()
            .filter_map(|(op_id, opt)| opt.map(|(ins, outs)| (op_id, ins, outs)))
            .collect();
        // Already sorted since we iterate by index

        // Two-pointer sweep to compute number of transfers inserted before each op
        let mut t_idx = 0usize;
        for (op_id, new_inputs, new_outputs) in remap_entries {
            while t_idx < transfer_positions.len() && transfer_positions[t_idx] <= op_id {
                t_idx += 1;
            }

            let adjusted_op_id = op_id + t_idx;
            self.tensor_graph.operations[adjusted_op_id]
                .remap_tensor_ids(&new_inputs, &new_outputs);
        }

        // Now that planning is complete, actually allocate the tensors
        self.allocate_tensors(tensor_locations, initialisers)?;

        Ok(())
    }

    fn allocate_tensors(
        &mut self,
        tensor_locations: Vec<Option<DeviceId>>,
        mut initialisers: Vec<Option<Box<[u8]>>>,
    ) -> Result<(), VKMLError> {
        let count = self.tensor_graph.tensor_descs.len();

        self.tensors.reserve(count);
        let out_ptr: *mut TensorCell = self.tensors.as_mut_ptr();

        let mut tasks: Vec<SingleAllocParams> = Vec::with_capacity(count);

        for i in 0..count {
            tasks.push(SingleAllocParams {
                index: i,
                initialisers_ptr: initialisers.as_mut_ptr(),
                manager_ptr: self as *const ComputeManager,
                out_ptrs: out_ptr,
                tensor_locations_ptr: tensor_locations.as_ptr(),
            });
        }

        global_pool()
            .submit_batch_uniform(single_allocate_task, &tasks)
            .wait();

        unsafe { self.tensors.set_len(count) };

        Ok(())
    }

    pub fn allocate_tensor(
        &self,
        desc: &TensorDesc,
        target_device: &DeviceId,
        init_box: Option<Box<[u8]>>,
    ) -> Result<Tensor, VKMLError> {
        let size_in_bytes = desc.size_in_bytes() as u64;

        match target_device {
            DeviceId::CPU => {
                self.cpu.memory_tracking.allocate(size_in_bytes);
                if let Some(boxed) = init_box {
                    if boxed.len() != desc.size_in_bytes() {
                        return Err(VKMLError::Generic(format!(
                            "Initialiser size mismatch for tensor: expected {} got {}",
                            desc.size_in_bytes(),
                            boxed.len()
                        )));
                    }
                    Ok(Tensor::new_cpu(desc.clone(), boxed))
                } else {
                    let buf = vec![0u8; size_in_bytes as usize];
                    Ok(Tensor::new_cpu(desc.clone(), buf.into()))
                }
            }
            DeviceId::GPU(idx) => {
                let gpu = &self.gpus.get_gpu(*idx);
                gpu.allocate_memory(size_in_bytes);

                if let Some(boxed) = init_box {
                    if boxed.len() != desc.size_in_bytes() {
                        return Err(VKMLError::Generic(format!(
                            "Initialiser size mismatch for tensor: expected {} got {}",
                            desc.size_in_bytes(),
                            boxed.len()
                        )));
                    }
                    let gpu_mem = gpu.move_to_gpu(&boxed);
                    Ok(Tensor::new_gpu(desc.clone(), *idx, gpu_mem))
                } else {
                    let gpu_mem = gpu
                        .allocate_uninitialised_gpu_memory(size_in_bytes as usize)
                        .map_err(|e| VKMLError::VulkanError(e.to_string()))?;
                    Ok(Tensor::new_gpu(desc.clone(), *idx, gpu_mem))
                }
            }
        }
    }

    pub fn forward(&mut self, batches: Vec<Tensor>) -> Result<Vec<Tensor>, VKMLError> {
        let input_tensor_ids = self.tensor_graph.get_input_tensor_ids();

        if batches.len() != input_tensor_ids.len() {
            return Err(VKMLError::VulkanError(format!(
                "Expected {} input batches, got {}",
                input_tensor_ids.len(),
                batches.len()
            )));
        }

        // Validate all sizes upfront
        for (batch_idx, batch) in batches.iter().enumerate() {
            let expected_bytes = self
                .tensor_read(input_tensor_ids[batch_idx])
                .desc
                .size_in_bytes();
            if batch.buffer.len_bytes() != expected_bytes {
                return Err(VKMLError::VulkanError(format!(
                    "Input batch {} size mismatch: got {} bytes, expected {} bytes",
                    batch_idx,
                    batch.buffer.len_bytes(),
                    expected_bytes
                )));
            }
        }

        // Load input data in parallel
        let load_params: Vec<_> = batches
            .into_iter()
            .enumerate()
            .map(|(batch_idx, batch)| BatchLoadParams {
                tensor_id: input_tensor_ids[batch_idx],
                batch,
                compute_manager: self,
            })
            .collect();

        global_pool()
            .submit_batch_uniform(batch_load_task, &load_params)
            .wait();

        self.execute()?;

        // Gather output data in parallel
        let output_tensor_ids = self.tensor_graph.get_output_tensor_ids();
        let output_count = output_tensor_ids.len();

        let mut output_batches: Vec<Tensor> = Vec::with_capacity(output_count);
        let out_ptr: *mut Tensor = output_batches.as_mut_ptr();

        let copy_params: Vec<_> = output_tensor_ids
            .iter()
            .enumerate()
            .map(|(idx, &tensor_id)| BatchCopyParams {
                tensor_id,
                output_index: idx,
                compute_manager: self,
                out_ptr,
            })
            .collect();

        global_pool()
            .submit_batch_uniform(batch_copy_task, &copy_params)
            .wait();

        unsafe { output_batches.set_len(output_count) };

        Ok(output_batches)
    }

    pub fn execute(&mut self) -> Result<(), VKMLError> {
        // Get or create cached execution plan
        if self.cached_execution_plan.is_none() {
            let plan = self.tensor_graph.create_execution_plan(&self.tensors);

            // Initialize command buffer cache with one entry per operation
            let op_count = self.tensor_graph.operations.len();
            self.cached_command_buffers = (0..op_count).map(|_| OnceLock::new()).collect();

            self.cached_execution_plan = Some(plan);
        }

        let plan = self.cached_execution_plan.as_ref().unwrap();

        // Allocate semaphore value offsets for each GPU (indexed by gpu_idx)
        let device_offsets: Vec<u64> = self
            .gpus
            .gpus()
            .iter()
            .enumerate()
            .map(|(gpu_idx, gpu)| {
                let device = DeviceId::GPU(gpu_idx);
                let count = plan.chunks.iter().filter(|c| c.device == device).count() as u64;
                if count > 0 {
                    gpu.allocate_semaphore_values(count)
                } else {
                    0
                }
            })
            .collect();

        // Execute chunks as thread pool tasks
        let chunk_params: Vec<_> = plan
            .chunks
            .iter()
            .map(|chunk| ChunkExecutionParams {
                chunk,
                compute_manager: self,
                device_offsets: &device_offsets,
            })
            .collect();

        global_pool()
            .submit_batch_uniform(chunk_execution_task, &chunk_params)
            .wait();

        // Wait for all GPU work to complete
        self.wait_for_gpu_completion(plan, &device_offsets)?;

        Ok(())
    }

    fn wait_for_gpu_completion(
        &self,
        plan: &ExecutionPlan,
        device_offsets: &[u64],
    ) -> Result<(), VKMLError> {
        let gpu_waits: Vec<(usize, u64)> = (0..self.gpus.gpus().len())
            .filter_map(|gpu_idx| {
                let offset = device_offsets[gpu_idx];
                plan.chunks
                    .iter()
                    .filter(|c| matches!(c.device, DeviceId::GPU(idx) if idx == gpu_idx))
                    .map(|c| c.signal_value + offset)
                    .max()
                    .map(|max_value| (gpu_idx, max_value))
            })
            .collect();

        if gpu_waits.is_empty() {
            return Ok(());
        }

        if gpu_waits.len() == 1 {
            let (gpu_idx, max_value) = gpu_waits[0];
            return self
                .gpus
                .get_gpu(gpu_idx)
                .wait_for_timeline_value(max_value);
        }

        // Multi-GPU: wait in parallel
        let wait_params: Vec<_> = gpu_waits
            .into_iter()
            .map(|(gpu_idx, semaphore_value)| GpuWaitParams {
                gpu_idx,
                semaphore_value,
                gpu_pool: &self.gpus,
            })
            .collect();

        global_pool()
            .submit_batch_uniform(gpu_wait_task, &wait_params)
            .wait();

        Ok(())
    }

    fn execute_gpu_chunk(&self, chunk: &ExecutionChunk, gpu_idx: usize, device_offsets: &[u64]) {
        let gpu = self.gpus.get_gpu(gpu_idx);

        unsafe {
            // Get or create cached command buffers for each operation in the chunk
            let command_buffers: Vec<_> = chunk
                .operations
                .iter()
                .map(|&op_id| {
                    *self.cached_command_buffers[op_id].get_or_init(|| {
                        // Allocate a single command buffer for this operation
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
                            .expect("Failed to allocate command buffer");

                        // Record command buffer for this operation
                        let instruction = self.tensor_graph.get_instruction_or_panic(op_id);
                        instruction
                            .create_command_buffer(gpu, buffers[0], self)
                            .expect("Failed to record command buffer");

                        buffers[0]
                    })
                })
                .collect();

            // Apply semaphore offset for this GPU
            let offset = device_offsets[gpu_idx];
            let signal_value = chunk.signal_value + offset;

            // Prepare wait semaphores
            let wait_sems: Vec<_> = chunk
                .wait_semaphores
                .iter()
                .filter_map(|(dev, val)| {
                    if let DeviceId::GPU(wait_gpu_idx) = dev {
                        let wait_offset = device_offsets[*wait_gpu_idx];
                        let wait_gpu = self.gpus.get_gpu(*wait_gpu_idx);
                        Some((
                            wait_gpu.get_or_create_timeline_semaphore(),
                            *val + wait_offset,
                        ))
                    } else {
                        None
                    }
                })
                .collect();

            // Submit with timeline semaphore
            if let Err(e) =
                gpu.submit_with_timeline_semaphore(&command_buffers, &wait_sems, signal_value)
            {
                eprintln!("Failed to submit GPU chunk: {}", e);
            }

            // Command buffers are cached and reused - never freed
        }
    }

    fn execute_cpu_chunk(&self, chunk: &ExecutionChunk, device_offsets: &[u64]) {
        // Wait for any GPU dependencies first (with offsets applied)
        for (wait_dev, wait_val) in &chunk.wait_semaphores {
            if let DeviceId::GPU(gpu_idx) = wait_dev {
                let offset = device_offsets[*gpu_idx];
                let gpu = self.gpus.get_gpu(*gpu_idx);
                if let Err(e) = gpu.wait_for_timeline_value(*wait_val + offset) {
                    eprintln!("Failed to wait for GPU {}: {}", gpu_idx, e);
                    return;
                }
            }
        }

        // Execute CPU operations in parallel
        let cpu_params: Vec<_> = chunk
            .operations
            .iter()
            .map(|&op_id| SingleCpuOperationParams {
                operation_id: op_id,
                compute_manager: self,
            })
            .collect();

        global_pool()
            .submit_batch_uniform(single_cpu_operation_task, &cpu_params)
            .wait();
    }

    pub fn format_memory_mb(&self, bytes: u64) -> String {
        format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
    }

    pub fn get_memory_usage_summary(&self) -> Vec<(String, String, String)> {
        let mut result = Vec::new();

        result.push((
            "CPU".to_string(),
            self.format_memory_mb(self.cpu.memory_tracking.get_current()),
            self.format_memory_mb(self.cpu.memory_tracking.get_available()),
        ));

        for (i, gpu) in self.gpus.gpus().iter().enumerate() {
            result.push((
                format!("GPU {}", i),
                self.format_memory_mb(gpu.total_memory() - gpu.available_memory()),
                self.format_memory_mb(gpu.available_memory()),
            ));
        }

        result
    }

    pub fn print_model_stats(&self) {
        print_model_stats::print_model_stats(self);
    }

    pub fn print_layer_values(&self, layer_id: LayerId) -> Result<(), VKMLError> {
        print_model_stats::print_layer_values(self, layer_id)
    }

    pub fn print_tensor_flow(&self) {
        print_tensorgraph_stats::print_tensor_flow(self);
    }

    pub fn tensor_read(&self, tensor_id: usize) -> &Tensor {
        unsafe { self.tensors[tensor_id].as_ref() }
    }

    // safety: uses UnsafeCell; scheduler guarantees exclusive mutable access
    #[allow(clippy::mut_from_ref)]
    pub fn tensor_write(&self, tensor_id: usize) -> &mut Tensor {
        unsafe { self.tensors[tensor_id].as_mut() }
    }
}

struct SingleAllocParams {
    index: usize,
    initialisers_ptr: *mut Option<Box<[u8]>>,
    manager_ptr: *const ComputeManager,
    out_ptrs: *mut TensorCell,
    tensor_locations_ptr: *const Option<DeviceId>,
}

zp_define_task_fn!(single_allocate_task, SingleAllocParams, |params| {
    let init_box = unsafe { (*params.initialisers_ptr.add(params.index)).take() };

    let manager: &ComputeManager = unsafe { &*params.manager_ptr };

    let desc: &TensorDesc = &manager.tensor_graph.tensor_descs[params.index];

    let target = unsafe {
        (*params.tensor_locations_ptr.add(params.index))
            .clone()
            .unwrap_or(DeviceId::CPU)
    };

    let tensor = manager.allocate_tensor(desc, &target, init_box).unwrap();

    unsafe {
        let slot = params.out_ptrs.add(params.index);
        ptr::write(slot, TensorCell::new(tensor));
    }
});

struct SingleCpuOperationParams<'a> {
    operation_id: OperationId,
    compute_manager: &'a ComputeManager,
}

zp_define_task_fn!(
    single_cpu_operation_task,
    SingleCpuOperationParams,
    |params| {
        let instruction = params
            .compute_manager
            .tensor_graph
            .get_instruction_or_panic(params.operation_id);
        instruction.execute_cpu(params.compute_manager);
    }
);

struct ChunkExecutionParams<'a> {
    chunk: &'a crate::tensor_graph::tensor_graph::ExecutionChunk,
    compute_manager: &'a ComputeManager,
    device_offsets: &'a [u64],
}

zp_define_task_fn!(chunk_execution_task, ChunkExecutionParams, |params| {
    match &params.chunk.device {
        DeviceId::GPU(gpu_idx) => {
            params
                .compute_manager
                .execute_gpu_chunk(params.chunk, *gpu_idx, params.device_offsets);
        }
        DeviceId::CPU => {
            params
                .compute_manager
                .execute_cpu_chunk(params.chunk, params.device_offsets);
        }
    }
});

struct GpuWaitParams<'a> {
    gpu_idx: usize,
    semaphore_value: u64,
    gpu_pool: &'a GpuPool,
}

zp_define_task_fn!(gpu_wait_task, GpuWaitParams, |params| {
    let gpu = params.gpu_pool.get_gpu(params.gpu_idx);
    if let Err(e) = gpu.wait_for_timeline_value(params.semaphore_value) {
        eprintln!("Failed to wait for GPU {}: {}", params.gpu_idx, e);
    }
});

struct BatchLoadParams<'a> {
    tensor_id: usize,
    batch: Tensor,
    compute_manager: &'a ComputeManager,
}

zp_define_task_fn!(batch_load_task, BatchLoadParams, |params| {
    params
        .compute_manager
        .tensor_write(params.tensor_id)
        .write(&params.batch.read());
});

struct BatchCopyParams<'a> {
    tensor_id: usize,
    output_index: usize,
    compute_manager: &'a ComputeManager,
    out_ptr: *mut Tensor,
}

zp_define_task_fn!(batch_copy_task, BatchCopyParams, |params| {
    let tensor = params.compute_manager.tensor_read(params.tensor_id);
    let output_data = tensor.read();
    let batch = Tensor::new_cpu(tensor.desc.clone(), output_data);

    unsafe {
        let slot = params.out_ptr.add(params.output_index);
        ptr::write(slot, batch);
    }
});
