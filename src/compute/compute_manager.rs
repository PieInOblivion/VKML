use std::collections::HashMap;
use std::sync::Arc;

use crate::dataloader::data_batch::DataBatch;
use crate::dataloader::dataloader::SourceFormat;
use crate::instruction::factory::Instructions;
use crate::instruction::instruction::Instruction;
use crate::tensor::compute_tensor::ComputeTensor;
use crate::tensor_graph::shared_tensor_graph::{SharedGPU, SharedTensorGraph};
use crate::tensor_graph::tensor_graph::{OperationId, TensorGraph};
use crate::thread_pool::worker::WorkType;
use crate::{
    dataloader::error::VKMLEngineError,
    gpu::vk_gpu::GPU,
    model::{graph_model::GraphModel, layer_connection::LayerId, weight_init::WeightInit},
    tensor::{tensor_data::TensorData, tensor_desc::TensorDesc},
    thread_pool::thread_pool::ThreadPool,
};

use super::cpu_compute::CPUCompute;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum DeviceLocation {
    CPU,
    GPU(usize),
}

pub struct ComputeManager {
    gpus: Vec<GPU>,
    cpu: CPUCompute,
    thread_pool: Arc<ThreadPool>,

    pub model: GraphModel,
    pub tensor_graph: TensorGraph,
}

impl ComputeManager {
    pub fn new(model: GraphModel, thread_pool: Arc<ThreadPool>) -> Result<Self, VKMLEngineError> {
        let gpus = Self::available_gpus()?;
        Self::new_with(model, thread_pool, gpus, None)
    }

    pub fn new_with(
        mut model: GraphModel,
        thread_pool: Arc<ThreadPool>,
        gpus: Vec<GPU>,
        cpu_memory_limit_bytes: Option<u64>,
    ) -> Result<Self, VKMLEngineError> {
        if model.verified.is_none() {
            model.verify()?;
        }

        let cpu = CPUCompute::new(cpu_memory_limit_bytes, thread_pool.clone());

        let tensor_graph = TensorGraph::from_graph_model(&model)?;

        let mut manager = Self {
            gpus,
            cpu,
            thread_pool,
            model,
            tensor_graph,
        };

        let total_memory = manager.tensor_graph.calculate_memory_requirements();
        let total_available: u64 = manager
            .gpus
            .iter()
            .map(|gpu| gpu.available_memory())
            .sum::<u64>()
            + manager.cpu.memory_tracking.get_available();

        if total_memory > total_available {
            return Err(VKMLEngineError::OutOfMemory(format!(
                "Model requires {} bytes but only {} available",
                total_memory, total_available
            )));
        }

        manager.allocate_tensors()?;

        Ok(manager)
    }

    fn available_gpus() -> Result<Vec<GPU>, VKMLEngineError> {
        let gpu_info = GPU::available_gpus()?;
        let mut gpus = Vec::with_capacity(gpu_info.len());

        for info in gpu_info {
            if let Ok(gpu) = GPU::new(info.device_index) {
                gpus.push(gpu);
            }
        }

        Ok(gpus)
    }

    // Requirements for this function as currently designed.
    // There's mostly two stages of optimisation for the flattened tensor graph
    // The execution plan which plans parrallel compute
    // and this tensor allocation stratagy.
    // They might become more intertwined in the future, but currently
    // any planned optimisations can be designed seperatly between the two.
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

    fn allocate_tensors(&mut self) -> Result<(), VKMLEngineError> {
        // Get execution plan and flatten to a linear sequence of operations
        let execution_plan = self.tensor_graph.create_execution_plan();
        let flattened_ops: Vec<OperationId> = execution_plan.into_iter().flatten().collect();

        // Track planned tensor locations: tensor_id -> DeviceLocation
        let mut tensor_locations: Vec<Option<DeviceLocation>> =
            vec![None; self.tensor_graph.tensors.len()];

        // Maintain a list of tensor remappings: original_id -> new_id on device
        let mut tensor_remappings: HashMap<(usize, DeviceLocation), usize> = HashMap::new();

        // Store remappings needed for operations: op_id -> (new_inputs, new_outputs)
        let mut operation_remappings: HashMap<OperationId, (Vec<usize>, Vec<usize>)> =
            HashMap::new();

        // New tensors created for transfers - including layer info
        let mut new_tensors: Vec<(TensorDesc, DeviceLocation, WeightInit, Option<LayerId>)> =
            Vec::new();

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
                    memory_needed +=
                        self.tensor_graph.tensors[tensor_id].desc.size_in_bytes() as u64;
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
                    let tensor_size =
                        self.tensor_graph.tensors[tensor_id].desc.size_in_bytes() as u64;
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
                        let original_tensor = &self.tensor_graph.tensors[tensor_id];

                        // Preserve the layer association
                        let original_layer_id = self.tensor_graph.tensor_to_layer[tensor_id];

                        // Plan the new tensor
                        let tensor_desc = original_tensor.desc.clone();
                        let tensor_size = tensor_desc.size_in_bytes() as u64;
                        let weight_init = WeightInit::Constant(0.0); // Will be filled by transfer

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
                        let transfer_instr = Instructions::transfer_to_device(
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
                    let tensor_size =
                        self.tensor_graph.tensors[tensor_id].desc.size_in_bytes() as u64;
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
                        let original_tensor = &self.tensor_graph.tensors[tensor_id];

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
        for (tensor_desc, device_location, weight_init, layer_id) in new_tensors {
            // Add the new tensor
            self.tensor_graph.tensors.push(ComputeTensor {
                desc: tensor_desc,
                data: TensorData::Unallocated, // Will be allocated later
            });

            // Add the same layer ID to maintain the connection
            self.tensor_graph.tensor_to_layer.push(layer_id);

            tensor_locations.push(Some(device_location));
        }

        // 2. Insert transfer operations into the instruction list
        let mut op_id_shift = 0;
        for (insert_before_op, transfer_instr) in transfer_operations {
            // Adjust for previous insertions
            let adjusted_pos = insert_before_op + op_id_shift;
            self.tensor_graph
                .operations
                .insert(adjusted_pos, transfer_instr);
            op_id_shift += 1;
        }

        // 3. Apply the tensor remappings to all operations
        for (&op_id, (new_inputs, new_outputs)) in &operation_remappings {
            // Adjust for inserted transfer operations
            let adjusted_op_id = op_id + op_id_shift;
            let instruction = &mut self.tensor_graph.operations[adjusted_op_id];
            instruction.remap_tensor_ids(&new_inputs, &new_outputs);
        }

        // 4. Now that planning is complete, actually allocate the tensors
        for tensor_id in 0..tensor_locations.len() {
            if tensor_id >= self.tensor_graph.tensors.len() {
                break;
            }

            if let Some(device_location) = &tensor_locations[tensor_id] {
                if !self.tensor_graph.tensors[tensor_id].data.is_allocated() {
                    // Get the tensor description first
                    let tensor_desc = self.tensor_graph.tensors[tensor_id].desc.clone();

                    // Get the appropriate weight initialization
                    let weight_init = self.get_weight_init_for_tensor(tensor_id);

                    // Allocate the tensor data
                    let tensor_data =
                        self.allocate_tensor(&tensor_desc, device_location, &weight_init)?;

                    // Update the tensor with the allocated data
                    self.tensor_graph.tensors[tensor_id].data = tensor_data;
                }
            }
        }

        Ok(())
    }

    // Helper method to determine weight initialization for a tensor
    fn get_weight_init_for_tensor(&self, tensor_id: usize) -> WeightInit {
        if let Some(layer_id) = self.tensor_graph.tensor_to_layer[tensor_id] {
            if let Some(layer) = self.model.layers.get(&layer_id) {
                if let Some(weight_init) = &layer.weight_init {
                    return weight_init.clone();
                }
            }
        }

        self.model.weight_init.clone()
    }

    fn allocate_tensor(
        &self,
        desc: &TensorDesc,
        target_device: &DeviceLocation,
        weight_init: &WeightInit,
    ) -> Result<TensorData, VKMLEngineError> {
        let size_in_bytes = desc.size_in_bytes() as u64;
        let parallel_threshold = 10000;
        let total_elements = desc.num_elements();

        let initial_data = {
            if total_elements < parallel_threshold {
                weight_init.init(desc, total_elements)
            } else {
                weight_init.par_init(
                    desc,
                    total_elements,
                    parallel_threshold,
                    self.thread_pool.clone(),
                )
            }
        };

        match *target_device {
            DeviceLocation::CPU => {
                self.cpu.memory_tracking.allocate(size_in_bytes)?;
                Ok(TensorData::new_cpu(initial_data))
            }
            DeviceLocation::GPU(idx) => {
                let gpu = &self.gpus[idx];
                gpu.allocate_memory(size_in_bytes)?;

                let gpu_memory = gpu
                    .move_to_gpu_as_f32(&initial_data)
                    .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;

                Ok(TensorData::new_gpu(idx, gpu_memory))
            }
        }
    }

    pub fn forward(&mut self, batches: Vec<DataBatch>) -> Result<Vec<DataBatch>, VKMLEngineError> {
        // Get input tensor indices
        let input_tensor_ids = &self.tensor_graph.input_tensors;

        // Validate input batch count
        if batches.len() != input_tensor_ids.len() {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Expected {} input batches, got {}",
                input_tensor_ids.len(),
                batches.len()
            )));
        }

        // Validate all batch sizes upfront
        for (batch_idx, batch) in batches.iter().enumerate() {
            let tensor_id = input_tensor_ids[batch_idx];
            let expected_size = self.tensor_graph.tensors[tensor_id].desc.num_elements();
            let data_size = batch.data.len() / batch.format.bytes_per_element();

            if data_size != expected_size {
                return Err(VKMLEngineError::VulkanLoadError(format!(
                    "Input batch {} size mismatch: got {}, expected {}",
                    batch_idx, data_size, expected_size
                )));
            }
        }

        // Load input data into tensors
        for (batch_idx, batch) in batches.into_iter().enumerate() {
            let tensor_id = input_tensor_ids[batch_idx];
            let data = batch.to_f32();
            self.tensor_graph.tensors[tensor_id]
                .data
                .update_data(data)?;
        }

        // Execute the model
        self.execute()?;

        // Gather output data and convert to DataBatch objects
        let output_tensor_ids = &self.tensor_graph.output_tensors;
        let mut output_batches = Vec::with_capacity(output_tensor_ids.len());

        for (idx, &tensor_id) in output_tensor_ids.iter().enumerate() {
            // Get output data
            let output_data = self.tensor_graph.tensors[tensor_id].data.get_data()?;
            let tensor = &self.tensor_graph.tensors[tensor_id];

            // Convert f32 data to bytes
            let mut bytes = Vec::with_capacity(output_data.len() * 4);
            for &value in &output_data {
                bytes.extend_from_slice(&value.to_le_bytes());
            }

            // Create DataBatch with appropriate metadata
            // This portion will become more advanced in the future.
            // Perhaps with statistics or something more
            let batch = DataBatch {
                data: bytes.into_boxed_slice(),
                samples_in_batch: self.model.batch_size,
                bytes_per_sample: tensor.desc.size_in_bytes(),
                format: SourceFormat::F32,
                labels: None,
                batch_number: idx,
            };

            output_batches.push(batch);
        }

        Ok(output_batches)
    }

    pub fn execute(&self) -> Result<(), VKMLEngineError> {
        // Get the execution plan from tensor graph
        let execution_plan = self.tensor_graph.create_execution_plan();
        let device_grouped_plan = self.group_operations_by_device(&execution_plan)?;

        // Create shared tensor graph pointer for worker threads
        let shared_tensor_graph = SharedTensorGraph {
            tensor_graph: &self.tensor_graph as *const _,
        };

        // Execute each stage sequentially, but operations within a stage in parallel
        for stage in device_grouped_plan {
            let mut futures = Vec::new();

            // Process each device's operations in this stage
            for per_device_ops in stage {
                if per_device_ops.is_empty() {
                    continue;
                }

                // Determine device for this group
                let device = self.determine_operation_device(&per_device_ops[0])?;

                match device {
                    DeviceLocation::GPU(idx) => {
                        // GPU operations - submit as batch to worker thread
                        let gpu_ref = &self.gpus[idx];

                        let shared_gpu = SharedGPU {
                            gpu: Some(gpu_ref as *const _),
                        };

                        let instruction_indices: Vec<usize> =
                            per_device_ops.iter().map(|op_id| *op_id).collect();

                        // Submit GPU batch to thread pool
                        futures.push(self.thread_pool.submit_work(WorkType::GpuBatchOperations {
                            operations: per_device_ops,
                            instruction_indices,
                            shared_gpu: Arc::new(shared_gpu),
                            shared_tensor_graph: Arc::new(shared_tensor_graph.clone()),
                        }));
                    }
                    DeviceLocation::CPU => {
                        // Submit individual CPU operations to thread pool
                        for &op_id in &per_device_ops {
                            futures.push(self.thread_pool.submit_work(
                                WorkType::SingleCpuOperation {
                                    operation_id: op_id,
                                    instruction_idx: op_id,
                                    shared_tensor_graph: Arc::new(shared_tensor_graph.clone()),
                                },
                            ));
                        }
                    }
                }
            }

            // Wait for all operations to complete
            for future in futures {
                future.wait();
            }
        }

        Ok(())
    }

    fn group_operations_by_device(
        &self,
        execution_plan: &[Vec<OperationId>],
    ) -> Result<Vec<Vec<Vec<OperationId>>>, VKMLEngineError> {
        let mut device_grouped_plan = Vec::with_capacity(execution_plan.len());

        // Process each stage of the execution plan
        for stage in execution_plan {
            let mut device_batches: HashMap<DeviceLocation, Vec<OperationId>> = HashMap::new();

            // Group operations by device
            for op_id in stage {
                let device = self.determine_operation_device(op_id)?;
                device_batches.entry(device).or_default().push(*op_id);
            }

            // Convert HashMap to Vec<Vec<OperationId>>
            let device_stage: Vec<Vec<OperationId>> = device_batches.into_values().collect();
            device_grouped_plan.push(device_stage);
        }

        Ok(device_grouped_plan)
    }

    fn determine_operation_device(
        &self,
        op_id: &OperationId,
    ) -> Result<DeviceLocation, VKMLEngineError> {
        // Get input tensor IDs for this operation
        let input_tensor_ids = self.tensor_graph.get_operation_inputs(*op_id);

        // Check input tensors to determine device
        for &tensor_id in &input_tensor_ids {
            match &self.tensor_graph.tensors[tensor_id].data {
                TensorData::CPU(_) => return Ok(DeviceLocation::CPU),
                TensorData::GPU { gpu_idx, .. } => return Ok(DeviceLocation::GPU(*gpu_idx)),
                TensorData::Unallocated => continue,
            }
        }

        // If no inputs have device info, check outputs
        let output_tensor_ids = self.tensor_graph.get_operation_outputs(*op_id);

        for &tensor_id in &output_tensor_ids {
            match &self.tensor_graph.tensors[tensor_id].data {
                TensorData::CPU(_) => return Ok(DeviceLocation::CPU),
                TensorData::GPU { gpu_idx, .. } => return Ok(DeviceLocation::GPU(*gpu_idx)),
                TensorData::Unallocated => continue,
            }
        }

        // If we can't determine from tensor data, return error
        Err(VKMLEngineError::VulkanLoadError(format!(
            "Operation {:?} has no tensors allocated to a device",
            op_id
        )))
    }

    pub fn format_memory_mb(&self, bytes: u64) -> String {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
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

    pub fn print_layer_values(&self, layer_id: LayerId) -> Result<(), VKMLEngineError> {
        crate::compute::print_model_stats::print_layer_values(self, layer_id)
    }

    pub fn print_tensor_flow(&self) {
        crate::compute::print_tensorgraph_stats::print_tensor_flow(self);
    }
}

impl Drop for ComputeManager {
    fn drop(&mut self) {
        // Used to transmute a reference to tensorgraph and gpu for worker threads
        // so manual drop was required because of 'static lifetime.
        // This will remain despite the change for now just incase.
        // Will remove in future when code changes are less breaking.
        {
            self.tensor_graph.tensors = Vec::new();
        }
        {
            self.gpus = Vec::new();
        }
    }
}
