use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::importers::onnx_parser::OnnxParser;
use crate::instruction;
use crate::tensor::tensor::{DeviceId, Tensor};
use onnx_extractor::OnnxModel;
use zero_pool::{ZeroPool, zp_define_task_fn};

use crate::instruction::instruction::Instruction;
use crate::tensor_graph::tensor_graph::{OperationId, TensorGraph};
use crate::{
    dataloader::error::VKMLError,
    gpu::vk_gpu::GPU,
    model::{graph_model::GraphModel, layer_connection::LayerId},
    tensor::desc::TensorDesc,
};
use vulkanalia::vk::DeviceV1_0;

use super::cpu_compute::CPUCompute;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum DeviceLocation {
    CPU,
    GPU(usize),
}

pub struct ComputeManager {
    gpus: Arc<Vec<GPU>>,
    cpu: CPUCompute,
    thread_pool: Arc<ZeroPool>,

    pub tensors: Vec<RwLock<Tensor>>,

    pub model: GraphModel,
    pub tensor_graph: TensorGraph,
}

impl ComputeManager {
    pub fn new(model: GraphModel, thread_pool: Arc<ZeroPool>) -> Result<Self, VKMLError> {
        let gpus = Self::available_gpus()?;
        Self::new_with(model, thread_pool, gpus, None)
    }

    pub fn new_with(
        mut model: GraphModel,
        thread_pool: Arc<ZeroPool>,
        gpus: Vec<GPU>,
        cpu_memory_limit_bytes: Option<u64>,
    ) -> Result<Self, VKMLError> {
        if model.verified.is_none() {
            model.verify()?;
        }

        let cpu = CPUCompute::new(cpu_memory_limit_bytes);

        let tensor_graph = TensorGraph::from_graph_model(&model)?;

        let mut manager = Self {
            gpus: Arc::new(gpus),
            cpu,
            thread_pool,
            tensors: Vec::new(),
            model,
            tensor_graph: tensor_graph,
        };

        let total_memory = manager.tensor_graph.memory_requirements as u64;
        let total_available: u64 = manager
            .gpus
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

    pub fn new_onnx(onnx_path: &str, thread_pool: Arc<ZeroPool>) -> Result<Self, VKMLError> {
        let onnx_model = OnnxModel::load_from_file(onnx_path).map_err(|e| {
            VKMLError::OnnxImporterError(format!(
                "Failed to load ONNX model from '{}': {}",
                onnx_path, e
            ))
        })?;

        let (tensor_graph, tensor_bytes) = OnnxParser::parse_onnx_model(onnx_model)?;

        let gpus = Self::available_gpus()?;
        Self::new_from_tensor_graph_with(tensor_graph, tensor_bytes, thread_pool, gpus, None)
    }

    /// Create ComputeManager from ONNX file with custom settings
    pub fn new_onnx_with(
        onnx_path: &str,
        thread_pool: Arc<ZeroPool>,
        gpus: Vec<GPU>,
        cpu_memory_limit_bytes: Option<u64>,
    ) -> Result<Self, VKMLError> {
        let onnx_model = OnnxModel::load_from_file(onnx_path).map_err(|e| {
            VKMLError::OnnxImporterError(format!(
                "Failed to load ONNX model from '{}': {}",
                onnx_path, e
            ))
        })?;

        let (tensor_graph, tensor_bytes) = OnnxParser::parse_onnx_model(onnx_model)?;

        Self::new_from_tensor_graph_with(
            tensor_graph,
            tensor_bytes,
            thread_pool,
            gpus,
            cpu_memory_limit_bytes,
        )
    }

    pub fn new_from_tensor_graph_with(
        tensor_graph: TensorGraph,
        tensor_bytes: Vec<Option<Box<[u8]>>>,
        thread_pool: Arc<ZeroPool>,
        gpus: Vec<GPU>,
        cpu_memory_limit_bytes: Option<u64>,
    ) -> Result<Self, VKMLError> {
        let cpu = CPUCompute::new(cpu_memory_limit_bytes);

        // TODO: Implement one type of model representation.
        // Placeholder minimal model until graph-only mode is supported
        let model = GraphModel::new(1);

        let mut manager = Self {
            gpus: Arc::new(gpus),
            cpu,
            thread_pool,
            tensors: Vec::new(),
            model,
            tensor_graph: tensor_graph,
        };

        let total_memory = manager.tensor_graph.memory_requirements as u64;
        let total_available: u64 = manager
            .gpus
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

    fn available_gpus() -> Result<Vec<GPU>, VKMLError> {
        let gpu_info = GPU::available_gpus();
        let mut gpus = Vec::with_capacity(gpu_info.len());

        for info in gpu_info {
            if let Ok(gpu) = GPU::new(info.device_index) {
                gpus.push(gpu);
            }
        }

        Ok(gpus)
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

    fn allocate_tensor_graph(
        &mut self,
        initialisers: Vec<Option<Box<[u8]>>>,
    ) -> Result<(), VKMLError> {
        // Get execution plan and flatten to a linear sequence of operations
        let execution_plan = self.tensor_graph.create_execution_plan();
        let flattened_ops: Vec<OperationId> = execution_plan.into_iter().flatten().collect();

        // Track planned tensor locations: tensor_id -> DeviceLocation
        let mut tensor_locations: Vec<Option<DeviceLocation>> = vec![None; self.tensors.len()];

        // Maintain a list of tensor remappings: original_id -> new_id on device
        let mut tensor_remappings: HashMap<(usize, DeviceLocation), usize> = HashMap::new();

        // Store remappings needed for operations: op_id -> (new_inputs, new_outputs)
        let mut operation_remappings: HashMap<OperationId, (Vec<usize>, Vec<usize>)> =
            HashMap::new();

        // New tensors created for transfers - including layer info
        let mut new_tensors: Vec<(
            TensorDesc,
            DeviceLocation,
            Box<dyn Instruction>,
            Option<LayerId>,
        )> = Vec::new();

        // Transfer operations to add
        let mut transfer_operations: Vec<(OperationId, Box<dyn Instruction>)> = Vec::new();

        // Track available memory per device
        let mut available_memory: Vec<(DeviceLocation, u64)> = Vec::new();
        for (idx, gpu) in self.gpus.iter().enumerate() {
            available_memory.push((DeviceLocation::GPU(idx), gpu.available_memory()));
        }
        available_memory.push((
            DeviceLocation::CPU,
            self.cpu.memory_tracking.get_available(),
        ));

        // Start with the first device (typically the most powerful GPU)
        let mut current_device_idx = 0;
        let mut current_device = available_memory[current_device_idx].0.clone();

        // Process operations in sequence
        let mut op_idx = 0;
        while op_idx < flattened_ops.len() {
            let op_id = flattened_ops[op_idx];
            let instruction = &self.tensor_graph.operations[op_id];

            // Get tensors required for this operation
            let input_tensors = instruction.get_input_tensor_ids();
            let output_tensors = instruction.get_output_tensor_ids();

            // Calculate memory needed for unallocated tensors
            let mut memory_needed = 0;
            for &tensor_id in input_tensors.iter().chain(output_tensors.iter()) {
                if tensor_locations[tensor_id].is_none() {
                    memory_needed += self.tensor_read(tensor_id).desc.size_in_bytes() as u64;
                }
            }

            // Check if we need to move to the next device
            if memory_needed > available_memory[current_device_idx].1 {
                // Move to next device
                current_device_idx = (current_device_idx + 1) % available_memory.len();
                current_device = available_memory[current_device_idx].0.clone();

                // Restart the check with new device
                continue;
            }

            // Prepare new input/output lists for remapping
            let mut new_inputs = Vec::with_capacity(input_tensors.len());
            let mut new_outputs = Vec::with_capacity(output_tensors.len());
            let mut remapping_needed = false;

            // Handle input tensors
            for &tensor_id in &input_tensors {
                let remapping_key = (tensor_id, current_device.clone());

                // Check if tensor is unallocated
                if tensor_locations[tensor_id].is_none() {
                    // Plan to allocate the tensor on current device
                    let tensor_size = self
                        .tensor_graph
                        .tensor_read(tensor_id)
                        .desc
                        .size_in_bytes() as u64;
                    tensor_locations[tensor_id] = Some(current_device.clone());
                    available_memory[current_device_idx].1 -= tensor_size;

                    // Use original tensor ID
                    new_inputs.push(tensor_id);
                }
                // Check if tensor is already on a different device
                else if tensor_locations[tensor_id] != Some(current_device.clone()) {
                    // Check if we already have a mapping for this tensor
                    if let Some(&mapped_id) = tensor_remappings.get(&remapping_key) {
                        new_inputs.push(mapped_id);
                        remapping_needed = true;
                    } else {
                        // Create a new tensor on current device
                        let new_tensor_id = self.tensor_graph.tensors.len() + new_tensors.len();
                        let original_tensor = self.tensor_graph.tensor_read(tensor_id);

                        // Preserve the layer association
                        let original_layer_id = self.tensor_graph.tensor_to_layer[tensor_id];

                        // Plan the new tensor
                        let tensor_desc = original_tensor.desc.clone();
                        let tensor_size = tensor_desc.size_in_bytes() as u64;
                        let weight_init = instruction::init_constant(0, vec![0, 0, 0, 0]); // Will be filled by transfer

                        // Reserve memory on current device
                        available_memory[current_device_idx].1 -= tensor_size;

                        // Add to new tensors list with layer info
                        new_tensors.push((
                            tensor_desc,
                            current_device.clone(),
                            weight_init,
                            original_layer_id,
                        ));

                        // Create transfer instruction
                        let src_device = tensor_locations[tensor_id].clone().unwrap();
                        let transfer_instr = instruction::transfer(
                            tensor_id,
                            new_tensor_id,
                            src_device,
                            current_device.clone(),
                        );

                        // Record the planned transfer
                        transfer_operations.push((op_id, transfer_instr));

                        // Record the remapping
                        tensor_remappings.insert(remapping_key, new_tensor_id);

                        // Use the new tensor ID
                        new_inputs.push(new_tensor_id);
                        remapping_needed = true;
                    }
                } else {
                    // Tensor already on current device
                    new_inputs.push(tensor_id);
                }
            }

            // Handle output tensors similarly
            for &tensor_id in &output_tensors {
                let remapping_key = (tensor_id, current_device.clone());

                // Check if tensor is unallocated
                if tensor_locations[tensor_id].is_none() {
                    // Plan to allocate the tensor on current device
                    let tensor_size = self
                        .tensor_graph
                        .tensor_read(tensor_id)
                        .desc
                        .size_in_bytes() as u64;
                    tensor_locations[tensor_id] = Some(current_device.clone());
                    available_memory[current_device_idx].1 -= tensor_size;

                    // Use original tensor ID
                    new_outputs.push(tensor_id);
                }
                // If output tensor is already allocated on different device, we need new tensor
                else if tensor_locations[tensor_id] != Some(current_device.clone()) {
                    // Check if we already have a mapping
                    if let Some(&mapped_id) = tensor_remappings.get(&remapping_key) {
                        new_outputs.push(mapped_id);
                        remapping_needed = true;
                    } else {
                        // Create a new tensor on current device
                        let new_tensor_id = self.tensor_graph.tensors.len() + new_tensors.len();
                        let original_tensor = self.tensor_graph.tensor_read(tensor_id);

                        // Preserve the layer association
                        let original_layer_id = self.tensor_graph.tensor_to_layer[tensor_id];

                        // Plan the new tensor
                        let tensor_desc = original_tensor.desc.clone();
                        let tensor_size = tensor_desc.size_in_bytes() as u64;
                        let weight_init = self.get_weight_init_for_tensor(tensor_id);

                        // Reserve memory on current device
                        available_memory[current_device_idx].1 -= tensor_size;

                        // Add to new tensors list with layer info
                        new_tensors.push((
                            tensor_desc,
                            current_device.clone(),
                            weight_init,
                            original_layer_id,
                        ));

                        // Record the remapping
                        tensor_remappings.insert(remapping_key, new_tensor_id);

                        // Use the new tensor ID
                        new_outputs.push(new_tensor_id);
                        remapping_needed = true;
                    }
                } else {
                    // Tensor already on current device
                    new_outputs.push(tensor_id);
                }
            }

            // Store the remapping information for later application
            if remapping_needed {
                operation_remappings.insert(op_id, (new_inputs, new_outputs));
            }

            // Move to next operation
            op_idx += 1;
        }

        // Now apply all of our planned changes

        // 1. Create all new tensors for transfers
        for (tensor_desc, device_location, _weight_init, layer_id) in new_tensors {
            // Add the new tensor
            self.tensor_graph
                .tensors
                .push(RwLock::new(Tensor::new_unallocated(tensor_desc)));

            // Add the same layer ID to maintain the connection
            self.tensor_graph.tensor_to_layer.push(layer_id);

            tensor_locations.push(Some(device_location));
        }

        // 2. Insert transfer operations into the instruction list
        let mut op_id_shift = 0;
        for (insert_before_op, transfer_instr) in transfer_operations {
            let adjusted_pos = insert_before_op + op_id_shift;
            // Preserve layer mapping for the new transfer op by copying the layer of the original op
            let layer_id = self.tensor_graph.operation_to_layer[insert_before_op];
            self.tensor_graph
                .operations
                .insert(adjusted_pos, transfer_instr);
            self.tensor_graph
                .operation_to_layer
                .insert(adjusted_pos, layer_id);
            op_id_shift += 1;
        }

        // 3. Apply the tensor remappings to all operations
        for (&op_id, (new_inputs, new_outputs)) in &operation_remappings {
            // Adjust for inserted transfer operations
            let adjusted_op_id = op_id + op_id_shift;
            self.tensor_graph.operations[adjusted_op_id]
                .remap_tensor_ids(&new_inputs, &new_outputs);
        }

        // 4. Now that planning is complete, actually allocate the tensors
        // TODO: This can surely be better...
        for tensor_id in 0..tensor_locations.len() {
            if tensor_id >= self.tensor_graph.tensors.len() {
                break;
            }

            if let Some(device_location) = &tensor_locations[tensor_id] {
                let tensor = self.tensor_graph.tensor_read(tensor_id);
                if tensor.device == DeviceId::Unallocated {
                    // Get the tensor description first
                    let tensor_desc = tensor.desc.clone();
                    drop(tensor);

                    // Get the appropriate weight initialization (already remapped)
                    let weight_init = self.get_weight_init_for_tensor(tensor_id);

                    // Allocate the tensor backing (no initialization yet)
                    let new_tensor = self.allocate_empty_tensor(&tensor_desc, device_location)?;

                    // Update the tensor with the allocated data so the instruction can find it
                    self.tensor_graph.tensors[tensor_id] = RwLock::new(new_tensor);

                    // Run the initialization instruction now that the tensor exists in the graph
                    match device_location {
                        DeviceLocation::CPU => {
                            // Synchronous CPU init; execute_cpu will be parallelised later
                            weight_init.execute_cpu(&self.tensor_graph);
                        }
                        DeviceLocation::GPU(idx) => {
                            let gpu = &self.gpus[*idx];

                            unsafe {
                                let alloc_info = vulkanalia::vk::CommandBufferAllocateInfo {
                                    s_type:
                                        vulkanalia::vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                                    next: std::ptr::null(),
                                    command_pool: gpu.get_command_pool(),
                                    level: vulkanalia::vk::CommandBufferLevel::PRIMARY,
                                    command_buffer_count: 1,
                                };

                                let command_buffers = match gpu
                                    .get_device()
                                    .allocate_command_buffers(&alloc_info)
                                {
                                    Ok(cbs) => cbs,
                                    Err(e) => {
                                        // rollback memory tracking and fail
                                        gpu.deallocate_memory(tensor_desc.size_in_bytes() as u64);
                                        return Err(VKMLError::VulkanLoadError(format!(
                                            "Failed to allocate command buffer: {:?}",
                                            e
                                        )));
                                    }
                                };

                                let cmd = command_buffers[0];

                                if let Err(e) = weight_init.create_command_buffer(gpu, cmd, &self) {
                                    gpu.get_device().free_command_buffers(
                                        gpu.get_command_pool(),
                                        &command_buffers,
                                    );
                                    gpu.deallocate_memory(tensor_desc.size_in_bytes() as u64);
                                    return Err(VKMLError::VulkanLoadError(format!(
                                        "GPU init create_command_buffer failed: {}",
                                        e
                                    )));
                                }

                                if let Err(e) = gpu.submit_command_buffers_and_wait(&[cmd]) {
                                    gpu.get_device().free_command_buffers(
                                        gpu.get_command_pool(),
                                        &command_buffers,
                                    );
                                    gpu.deallocate_memory(tensor_desc.size_in_bytes() as u64);
                                    return Err(VKMLError::VulkanLoadError(format!(
                                        "Failed to submit init command buffer: {}",
                                        e
                                    )));
                                }

                                gpu.get_device()
                                    .free_command_buffers(gpu.get_command_pool(), &command_buffers);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn get_weight_init_for_tensor(&self, tensor_id: usize) -> Box<dyn Instruction> {
        if let Some(layer_id) = self.tensor_graph.tensor_to_layer[tensor_id] {
            if let Some(layer) = self.model.layers.get(&layer_id) {
                if let Some(weight_init) = &layer.weight_init {
                    let mut copy = weight_init.clone();
                    copy.remap_tensor_ids(&Vec::new(), &vec![tensor_id]);
                    return copy;
                }
            }
        }

        let mut copy = self.model.weight_init.clone();
        copy.remap_tensor_ids(&Vec::new(), &vec![tensor_id]);
        return copy;
    }

    fn allocate_empty_tensor(
        &self,
        desc: &TensorDesc,
        target_device: &DeviceLocation,
    ) -> Result<Tensor, VKMLError> {
        let size_in_bytes = desc.size_in_bytes() as u64;

        match target_device {
            DeviceLocation::CPU => {
                // Allocate zeroed host backing (initialisation will run via execute_cpu)
                let buf = vec![0u8; size_in_bytes as usize];
                self.cpu.memory_tracking.allocate(size_in_bytes);
                Ok(Tensor::new_cpu(desc.clone(), buf.into()))
            }
            DeviceLocation::GPU(idx) => {
                let gpu_idx = *idx; // copy the usize
                let gpu = &self.gpus[gpu_idx];

                // Reserve tracking
                gpu.allocate_memory(size_in_bytes);

                // Allocate uninitialised GPU memory (function takes usize bytes)
                let gpu_mem = gpu
                    .allocate_uninitialised_gpu_memory(size_in_bytes as usize)
                    .map_err(|e| VKMLError::VulkanLoadError(e.to_string()))?;

                Ok(Tensor::new_gpu(desc.clone(), gpu_idx, gpu_mem))
            }
        }
    }

    pub fn forward(&mut self, batches: Vec<Tensor>) -> Result<Vec<Tensor>, VKMLError> {
        // Get input tensor indices
        let input_tensor_ids = self.tensor_graph.get_input_tensor_ids();

        // Validate input batch count
        if batches.len() != input_tensor_ids.len() {
            return Err(VKMLError::VulkanLoadError(format!(
                "Expected {} input batches, got {}",
                input_tensor_ids.len(),
                batches.len()
            )));
        }

        // Load input data into tensors
        for (batch_idx, batch) in batches.into_iter().enumerate() {
            let tensor_id = input_tensor_ids[batch_idx];

            // Validate size after conversion
            let expected_bytes = {
                let tensor = self.tensor_read(tensor_id);
                tensor.desc.size_in_bytes()
            };

            if batch.buffer.len_bytes() != expected_bytes {
                return Err(VKMLError::VulkanLoadError(format!(
                    "Input batch {} size mismatch: got {} bytes, expected {} bytes",
                    batch_idx,
                    batch.buffer.len_bytes(),
                    expected_bytes
                )));
            }

            self.tensor_write(tensor_id).write(&batch.read());
        }

        // Execute the model
        self.execute()?;

        // Gather output data and convert to DataBatch objects
        let output_tensor_ids = &self.tensor_graph.get_output_tensor_ids();
        let mut output_batches = Vec::with_capacity(output_tensor_ids.len());

        for &tensor_id in output_tensor_ids.iter() {
            let tensor = self.tensor_read(tensor_id);

            // Get data from tensor (currently returns f32, but will return native type in future)
            let output_data = tensor.read();

            // For now, get_data() returns f32, so we need to convert to bytes
            // In future, get_data() will return bytes in the tensor's native format
            let mut bytes = Vec::with_capacity(output_data.len() * 4);
            for &value in &output_data {
                bytes.extend_from_slice(&value.to_le_bytes());
            }

            // Create DataBatch with tensor's data type
            // No conversion - just packaging the tensor's data with its type
            let batch = Tensor::new_cpu(tensor.desc.clone(), bytes.into());

            output_batches.push(batch);
        }

        Ok(output_batches)
    }

    pub fn execute(&self) -> Result<(), VKMLError> {
        let execution_plan = self.tensor_graph.create_execution_plan();
        let device_grouped_plan = self.group_operations_by_device(&execution_plan)?;

        // Execute each stage sequentially, but operations within a stage in parallel
        for stage in device_grouped_plan {
            let mut futures = Vec::new();
            // Keep heap-allocated task params alive until we wait on futures.
            let mut pending_gpu_task_boxes: Vec<Box<GpuBatchOperationsParams>> = Vec::new();
            let mut pending_cpu_task_boxes: Vec<Box<SingleCpuOperationParams>> = Vec::new();

            // Process each device's operations in this stage
            for per_device_ops in stage {
                if per_device_ops.is_empty() {
                    continue;
                }

                let device = self.determine_operation_device(&per_device_ops[0])?;

                match device {
                    DeviceLocation::GPU(idx) => {
                        let instruction_indices: Vec<usize> =
                            per_device_ops.iter().map(|op_id| *op_id).collect();

                        let boxed_task = Box::new(GpuBatchOperationsParams {
                            operations: per_device_ops,
                            instruction_indices,
                            gpu_idx: idx,
                            compute_manager: &self,
                        });

                        pending_gpu_task_boxes.push(boxed_task);
                        let task_ref: &GpuBatchOperationsParams =
                            pending_gpu_task_boxes.last().unwrap().as_ref();

                        futures.push(
                            self.thread_pool
                                .submit_task(gpu_batch_operations_task, task_ref),
                        );
                    }
                    DeviceLocation::CPU => {
                        // Submit each CPU operation as its own task so they can execute
                        // concurrently. Box each task param and store it in
                        // `pending_cpu_task_boxes` so the reference passed into the
                        // thread pool remains valid until we wait on the futures.
                        for &operation_id in &per_device_ops {
                            let boxed = Box::new(SingleCpuOperationParams {
                                operation_id,
                                compute_manager: &self,
                            });

                            pending_cpu_task_boxes.push(boxed);
                            let task_ref: &SingleCpuOperationParams =
                                pending_cpu_task_boxes.last().unwrap().as_ref();

                            futures.push(
                                self.thread_pool
                                    .submit_task(single_cpu_operation_task, task_ref),
                            );
                        }
                    }
                }
            }

            for future in futures {
                future.wait();
            }

            // Drop boxed tasks for this stage now that futures have completed
            drop(pending_gpu_task_boxes);
            drop(pending_cpu_task_boxes);
        }

        Ok(())
    }

    fn group_operations_by_device(
        &self,
        execution_plan: &[Vec<OperationId>],
    ) -> Result<Vec<Vec<Vec<OperationId>>>, VKMLError> {
        let mut device_grouped_plan = Vec::with_capacity(execution_plan.len());

        for stage in execution_plan {
            let mut device_batches: HashMap<DeviceLocation, Vec<OperationId>> = HashMap::new();

            // Group operations by device
            for op_id in stage {
                let device = self.determine_operation_device(op_id)?;
                device_batches.entry(device).or_default().push(*op_id);
            }

            let device_stage: Vec<Vec<OperationId>> = device_batches.into_values().collect();
            device_grouped_plan.push(device_stage);
        }

        Ok(device_grouped_plan)
    }

    fn determine_operation_device(&self, op_id: &OperationId) -> Result<DeviceLocation, VKMLError> {
        let input_tensor_ids = self.tensor_graph.get_operation_inputs(*op_id);

        // Check input tensors to determine device
        for &tensor_id in &input_tensor_ids {
            match self.tensor_read(tensor_id).device {
                DeviceId::CPU => return Ok(DeviceLocation::CPU),
                DeviceId::GPU(gpu_idx) => {
                    return Ok(DeviceLocation::GPU(gpu_idx));
                }
            }
        }

        // If no inputs have device info, check outputs
        let output_tensor_ids = self.tensor_graph.get_operation_outputs(*op_id);

        for &tensor_id in &output_tensor_ids {
            match self.tensor_read(tensor_id).device {
                DeviceId::CPU => return Ok(DeviceLocation::CPU),
                DeviceId::GPU(gpu_idx) => {
                    return Ok(DeviceLocation::GPU(gpu_idx));
                }
            }
        }

        Err(VKMLError::VulkanLoadError(format!(
            "Operation {:?} has no tensors allocated to a device",
            op_id
        )))
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

        for (i, gpu) in self.gpus.iter().enumerate() {
            result.push((
                format!("GPU {}", i),
                self.format_memory_mb(gpu.total_memory() - gpu.available_memory()),
                self.format_memory_mb(gpu.available_memory()),
            ));
        }

        result
    }

    pub fn print_model_stats(&self) {
        crate::compute::print_model_stats::print_model_stats(self);
    }

    pub fn print_layer_values(&self, layer_id: LayerId) -> Result<(), VKMLError> {
        crate::compute::print_model_stats::print_layer_values(self, layer_id)
    }

    pub fn print_tensor_flow(&self) {
        crate::compute::print_tensorgraph_stats::print_tensor_flow(self);
    }

    pub fn tensor_read(&self, tensor_id: usize) -> RwLockReadGuard<'_, Tensor> {
        self.tensors[tensor_id].read().unwrap()
    }

    pub fn tensor_write(&self, tensor_id: usize) -> RwLockWriteGuard<'_, Tensor> {
        self.tensors[tensor_id].write().unwrap()
    }
}

struct SingleCpuOperationParams<'a> {
    operation_id: OperationId,
    compute_manager: &'a ComputeManager,
}

zp_define_task_fn!(
    single_cpu_operation_task,
    SingleCpuOperationParams,
    |params| {
        let instruction = &params
            .compute_manager
            .tensor_graph
            .get_instruction_or_panic(params.operation_id);
        instruction.execute_cpu(&params.compute_manager);
    }
);

struct GpuBatchOperationsParams<'a> {
    operations: Vec<OperationId>,
    instruction_indices: Vec<usize>,
    gpu_idx: usize,
    compute_manager: &'a ComputeManager,
}

zp_define_task_fn!(
    gpu_batch_operations_task,
    GpuBatchOperationsParams,
    |params| {
        let gpu = &params.compute_manager.gpus[params.gpu_idx];

        if params.operations.is_empty() {
            return;
        }

        unsafe {
            let alloc_info = vulkanalia::vk::CommandBufferAllocateInfo {
                s_type: vulkanalia::vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                next: std::ptr::null(),
                command_pool: gpu.get_command_pool(),
                level: vulkanalia::vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: params.operations.len() as u32,
            };

            let Ok(command_buffers) = gpu.get_device().allocate_command_buffers(&alloc_info) else {
                eprintln!("Failed to allocate command buffers");
                return;
            };

            if command_buffers.len() != params.operations.len() {
                eprintln!("Mismatch between allocated command buffers and operations");
                gpu.get_device()
                    .free_command_buffers(gpu.get_command_pool(), &command_buffers);
                return;
            }

            let mut valid_buffers = Vec::new();

            for i in 0..params.operations.len() {
                let instruction = params
                    .compute_manager
                    .tensor_graph
                    .get_instruction_or_panic(params.instruction_indices[i]);

                if instruction
                    .create_command_buffer(&gpu, command_buffers[i], &params.compute_manager)
                    .is_ok()
                {
                    valid_buffers.push(command_buffers[i]);
                } else {
                    eprintln!(
                        "Failed to record command buffer for operation {:?}",
                        &params.operations[i]
                    );
                }
            }

            if !valid_buffers.is_empty() {
                if let Err(e) = gpu.submit_command_buffers_and_wait(&valid_buffers) {
                    eprintln!("Failed to submit batch operations: {}", e);
                }
            }

            gpu.get_device()
                .free_command_buffers(gpu.get_command_pool(), &command_buffers);
        }
    }
);
