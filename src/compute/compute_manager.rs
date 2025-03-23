use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::dataloader::data_batch::DataBatch;
use crate::dataloader::dataloader::SourceFormat;
use crate::tensor::compute_tensor::ComputeTensor;
use crate::tensor_graph::shared_tensor_graph::{SharedGPU, SharedTensorGraph};
use crate::tensor_graph::tensor_graph::{OperationId, TensorGraph, TensorId};
use crate::thread_pool::worker::WorkType;
use crate::{
    dataloader::error::VKMLEngineError,
    gpu::vk_gpu::GPU,
    model::{
        graph_model::GraphModel, instruction::Instruction, layer_connection::LayerId,
        weight_init::WeightInit,
    },
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
    fn allocate_tensors(&mut self) -> Result<(), VKMLEngineError> {
        // Get execution plan and flatten to a linear sequence of operations
        let execution_plan = self.tensor_graph.create_execution_plan();
        let operations: Vec<OperationId> = execution_plan.into_iter().flatten().collect();

        // device tracking
        let mut available_devices = Vec::new();
        for (idx, gpu) in self.gpus.iter().enumerate() {
            available_devices.push((DeviceLocation::GPU(idx), gpu.available_memory()));
        }
        available_devices.push((
            DeviceLocation::CPU,
            self.cpu.memory_tracking.get_available(),
        ));

        let mut tensor_device_map: HashMap<TensorId, DeviceLocation> = HashMap::new();

        // Pre-compute where each input tensor is first used
        let mut input_first_use_op: HashMap<TensorId, usize> = HashMap::new();

        for (op_idx, op_id) in operations.iter().enumerate() {
            if let Some(inputs) = self.tensor_graph.operation_inputs.get(op_id) {
                for input_id in inputs {
                    if self.tensor_graph.input_tensors.contains(input_id)
                        && !input_first_use_op.contains_key(input_id)
                    {
                        input_first_use_op.insert(input_id.clone(), op_idx);
                    }
                }
            }
        }

        // Process operations in execution order
        let mut op_idx = 0;
        let mut current_device_idx = 0;

        while op_idx < operations.len() {
            // Process operations until current device is full
            while op_idx < operations.len() {
                let op_id = &operations[op_idx];

                // Build a set of tensors to allocate
                let mut tensors_to_allocate = HashSet::new();

                // Regular operation tensors
                if let Some(inputs) = self.tensor_graph.operation_inputs.get(op_id) {
                    tensors_to_allocate.extend(inputs.iter().cloned());
                }

                if let Some(outputs) = self.tensor_graph.operation_outputs.get(op_id) {
                    tensors_to_allocate.extend(outputs.iter().cloned());
                }

                // Include any input tensors that are first used by this operation
                for (input_id, first_op_idx) in &input_first_use_op {
                    if *first_op_idx == op_idx {
                        tensors_to_allocate.insert(input_id.clone());
                    }
                }

                // Calculate memory needed for unallocated tensors
                let memory_needed: u64 = tensors_to_allocate
                    .iter()
                    .filter(|tensor_id| !tensor_device_map.contains_key(tensor_id))
                    .map(|tensor_id| {
                        let tensor = self
                            .tensor_graph
                            .get_tensor_by_id_or_error(tensor_id)
                            .expect("Tensor in operation not found in tensor graph");
                        tensor.desc.size_in_bytes() as u64
                    })
                    .sum();

                // Check if this operation fits on current device
                if memory_needed > available_devices[current_device_idx].1 {
                    break; // Device full, move to next
                }

                // Clone the current device to avoid borrowing conflicts
                let current_device_type = available_devices[current_device_idx].0.clone();

                for tensor_id in &tensors_to_allocate {
                    if !tensor_device_map.contains_key(tensor_id) {
                        let tensor = self
                            .tensor_graph
                            .get_tensor_by_id_or_error(tensor_id)
                            .expect("Tensor in operation not found in tensor graph");
                        let size = tensor.desc.size_in_bytes() as u64;

                        tensor_device_map.insert(tensor_id.clone(), current_device_type.clone());
                        available_devices[current_device_idx].1 -= size;
                    }
                }

                op_idx += 1;
            }

            if op_idx >= operations.len() {
                break;
            }

            let next_device_idx = current_device_idx + 1;

            if next_device_idx >= available_devices.len() {
                return Err(VKMLEngineError::OutOfMemory(
                    "Model requires more memory than available across all devices".to_string(),
                ));
            }

            let current_device_type = available_devices[current_device_idx].0.clone();
            let next_device_type = available_devices[next_device_idx].0.clone();

            let mut tensors_to_transfer = HashSet::new();

            for future_op_idx in op_idx..operations.len() {
                let future_op = &operations[future_op_idx];

                if let Some(inputs) = self.tensor_graph.operation_inputs.get(future_op) {
                    for tensor_id in inputs {
                        if tensor_device_map.get(tensor_id) == Some(&current_device_type) {
                            tensors_to_transfer.insert(tensor_id.clone());
                        }
                    }
                }
            }

            // Create transfers for each tensor
            for tensor_id in tensors_to_transfer {
                let storage_id = self.create_storage_tensor_id(&tensor_id, &next_device_type);

                let tensor = self.tensor_graph.get_tensor_by_id_or_error(&tensor_id)?;
                let size = tensor.desc.size_in_bytes() as u64;

                if size > available_devices[next_device_idx].1 {
                    return Err(VKMLEngineError::OutOfMemory(
                        "Not enough memory on next device for tensor transfer".to_string(),
                    ));
                }

                // Create new tensor in graph if needed
                if !self.tensor_graph.tensors.contains_key(&storage_id) {
                    let new_tensor = ComputeTensor {
                        desc: tensor.desc.clone(),
                        data: TensorData::Unallocated,
                    };
                    self.tensor_graph
                        .tensors
                        .insert(storage_id.clone(), new_tensor);
                }

                tensor_device_map.insert(storage_id.clone(), next_device_type.clone());
                available_devices[next_device_idx].1 -= size;

                self.create_transfer_operation(
                    &tensor_id,
                    &storage_id,
                    &current_device_type,
                    &next_device_type,
                )?;

                self.update_future_operations(&operations[op_idx..], &tensor_id, &storage_id)?;
            }

            current_device_idx = next_device_idx;
        }

        // Check for any truly unused input tensors
        let unused_inputs: Vec<_> = self
            .tensor_graph
            .input_tensors
            .iter()
            .filter(|id| !tensor_device_map.contains_key(*id))
            .collect();

        if !unused_inputs.is_empty() {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Input tensors {:?} are not used by any operation",
                unused_inputs
            )));
        }

        // Perform physical allocation of all tensors
        for (tensor_id, device) in &tensor_device_map {
            let tensor = self.tensor_graph.get_tensor_by_id_or_error(tensor_id)?;

            if let TensorData::Unallocated = tensor.data {
                let weight_init = self
                    .model
                    .layers
                    .get(&tensor_id.0)
                    .and_then(|layer| layer.weight_init.as_ref())
                    .unwrap_or(&self.model.weight_init);

                let allocated_data = self.allocate_tensor(&tensor.desc, device, weight_init)?;

                if let Some(tensor) = self.tensor_graph.get_tensor_mut(tensor_id.0, &tensor_id.1) {
                    tensor.data = allocated_data;
                }
            }
        }

        Ok(())
    }

    fn create_storage_tensor_id(
        &self,
        tensor_id: &TensorId,
        target_device: &DeviceLocation,
    ) -> TensorId {
        let device_suffix = match target_device {
            DeviceLocation::CPU => "_cpu".to_string(),
            DeviceLocation::GPU(idx) => format!("_gpu{}", idx),
        };

        TensorId(tensor_id.0, format!("{}{}", tensor_id.1, device_suffix))
    }

    fn create_transfer_operation(
        &mut self,
        src_id: &TensorId,
        dst_id: &TensorId,
        src_device: &DeviceLocation,
        dst_device: &DeviceLocation,
    ) -> Result<(), VKMLEngineError> {
        // Create a new operation ID
        let max_op_id = self
            .tensor_graph
            .operations
            .keys()
            .map(|op| op.1)
            .max()
            .unwrap_or(0);

        let op_id = OperationId(src_id.0, max_op_id + 1);

        let transfer_op = Instruction::TransferToDevice {
            src: src_id.1.clone(),
            dst: dst_id.1.clone(),
            source_device: src_device.clone(),
            target_device: dst_device.clone(),
        };

        self.tensor_graph
            .operations
            .insert(op_id.clone(), transfer_op);

        let mut inputs = HashSet::new();
        inputs.insert(src_id.clone());
        self.tensor_graph
            .operation_inputs
            .insert(op_id.clone(), inputs);

        let mut outputs = HashSet::new();
        outputs.insert(dst_id.clone());
        self.tensor_graph
            .operation_outputs
            .insert(op_id.clone(), outputs);

        self.tensor_graph
            .tensor_dependencies
            .entry(dst_id.clone())
            .or_insert_with(HashSet::new)
            .insert(op_id.clone());

        if !self.tensor_graph.tensors.contains_key(dst_id) {
            let src_tensor = self.tensor_graph.get_tensor_by_id_or_error(src_id)?;
            let new_tensor = ComputeTensor {
                desc: src_tensor.desc.clone(),
                data: TensorData::Unallocated,
            };
            self.tensor_graph.tensors.insert(dst_id.clone(), new_tensor);
        }

        Ok(())
    }

    fn update_future_operations(
        &mut self,
        future_ops: &[OperationId],
        original_id: &TensorId,
        storage_id: &TensorId,
    ) -> Result<(), VKMLEngineError> {
        for op_id in future_ops {
            if let Some(inputs) = self.tensor_graph.operation_inputs.get_mut(op_id) {
                if inputs.contains(original_id) {
                    inputs.remove(original_id);
                    inputs.insert(storage_id.clone());
                }
            }
        }

        Ok(())
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
        let input_tensor_ids = self.tensor_graph.input_tensors.clone();

        // Requires all inputs exist
        if batches.len() != input_tensor_ids.len() {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Expected {} input batches, got {}",
                input_tensor_ids.len(),
                batches.len()
            )));
        }

        // Validate all batch sizes upfront before any processing
        for (batch_idx, batch) in batches.iter().enumerate() {
            let tensor_id = &input_tensor_ids[batch_idx];

            let expected_size = self
                .tensor_graph
                .get_tensor_by_id_or_error(tensor_id)?
                .desc
                .num_elements();

            let data_size = batch.data.len() / batch.format.bytes_per_element();
            if data_size != expected_size {
                return Err(VKMLEngineError::VulkanLoadError(format!(
                    "Input batch {} size mismatch: got {}, expected {}",
                    batch_idx, data_size, expected_size
                )));
            }
        }

        for (batch_idx, batch) in batches.into_iter().enumerate() {
            let tensor_id = &input_tensor_ids[batch_idx];
            let data = batch.to_f32();
            self.tensor_graph
                .get_tensor_by_id_or_error(tensor_id)?
                .data
                .update_data(data)?;
        }

        self.execute()?;

        // Gather output data and convert to DataBatch objects
        let output_tensor_ids = self.tensor_graph.get_output_tensor_ids();
        let mut output_batches = Vec::with_capacity(output_tensor_ids.len());

        for (idx, output_id) in output_tensor_ids.iter().enumerate() {
            let output_data = self
                .tensor_graph
                .get_tensor_by_id_or_error(output_id)?
                .data
                .get_data()?;

            // determine output format
            let tensor = self.tensor_graph.get_tensor_by_id_or_error(output_id)?;

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
        let execution_plan = self.tensor_graph.create_execution_plan();
        let device_grouped_plan = self.group_operations_by_device(&execution_plan)?;

        // Create a SharedTensorGraph with a raw pointer to tensor_graph
        // This is identical for all operations
        let shared_tensor_graph = SharedTensorGraph {
            tensor_graph: &self.tensor_graph as *const _,
        };

        for stage in device_grouped_plan {
            let mut device_futures = Vec::new();

            for per_device_steps in stage {
                let device = self.determine_operation_device(&per_device_steps[0])?;

                let gpu_ref = match &device {
                    DeviceLocation::GPU(idx) => Some(&self.gpus[*idx]),
                    DeviceLocation::CPU => None,
                };

                // Create a SharedGPU with a raw pointer to the GPU
                let shared_gpu = SharedGPU {
                    gpu: gpu_ref.map(|gpu| gpu as *const _),
                };

                // Wrap both in Arc to share across threads
                let shared_gpu_arc = Arc::new(shared_gpu);
                let shared_graph_arc = Arc::new(shared_tensor_graph.clone());

                let operations = per_device_steps.clone();
                let instructions: Vec<Instruction> = operations
                    .iter()
                    .filter_map(|op_id| self.tensor_graph.operations.get(op_id).cloned())
                    .collect();

                device_futures.push(WorkType::TensorOperations {
                    operations,
                    instructions,
                    shared_gpu: shared_gpu_arc,
                    shared_tensor_graph: shared_graph_arc,
                });
            }

            let futures = self.thread_pool.submit_batch(device_futures);

            futures.wait();
        }

        Ok(())
    }

    fn group_operations_by_device(
        &self,
        execution_plan: &[Vec<OperationId>],
    ) -> Result<Vec<Vec<Vec<OperationId>>>, VKMLEngineError> {
        let mut device_grouped_plan = Vec::with_capacity(execution_plan.len());

        for stage in execution_plan {
            let mut device_batches: HashMap<DeviceLocation, Vec<OperationId>> = HashMap::new();

            for op_id in stage {
                let device = self.determine_operation_device(op_id)?;

                device_batches
                    .entry(device)
                    .or_insert_with(Vec::new)
                    .push(op_id.clone());
            }

            // Convert HashMap to Vec<Vec<OperationId>> (order doesn't matter within a device batch)
            let device_stage: Vec<Vec<OperationId>> = device_batches.into_values().collect();

            device_grouped_plan.push(device_stage);
        }

        Ok(device_grouped_plan)
    }

    fn determine_operation_device(
        &self,
        op_id: &OperationId,
    ) -> Result<DeviceLocation, VKMLEngineError> {
        if let Some(instruction) = self.tensor_graph.operations.get(op_id) {
            if let Instruction::TransferToDevice { source_device, .. } = instruction {
                return Ok(source_device.clone());
            }
        }

        let input_tensor_ids = self.tensor_graph.get_operation_inputs_or_error(op_id)?;

        // For non-transfer operations, check tensors to determine device
        // Start with input tensors which must exist for execution
        for tensor_id in input_tensor_ids {
            let tensor = self.tensor_graph.get_tensor_by_id_or_error(tensor_id)?;
            match &tensor.data {
                TensorData::CPU(_) => return Ok(DeviceLocation::CPU),
                TensorData::GPU { gpu_idx, .. } => return Ok(DeviceLocation::GPU(*gpu_idx)),
                TensorData::Unallocated => continue,
            }
        }

        let output_tensor_ids = self.tensor_graph.get_operation_outputs_or_error(op_id)?;

        for tensor_id in output_tensor_ids {
            let tensor = self.tensor_graph.get_tensor_by_id_or_error(tensor_id)?;
            match &tensor.data {
                TensorData::CPU(_) => return Ok(DeviceLocation::CPU),
                TensorData::GPU { gpu_idx, .. } => return Ok(DeviceLocation::GPU(*gpu_idx)),
                TensorData::Unallocated => continue,
            }
        }

        // If we got here, no tensors are allocated to a device
        Err(VKMLEngineError::VulkanLoadError(format!(
            "Operation {:?} has no tensors allocated to a device",
            op_id
        )))
    }

    fn set_tensor_data(&self, tensor_id: &TensorId, data: Vec<f32>) -> Result<(), VKMLEngineError> {
        let tensor = self.tensor_graph.get_tensor_by_id_or_error(tensor_id)?;
        tensor.data.update_data(data)
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

    pub fn get_device_description(&self, tensor_id: &TensorId) -> String {
        if let Some(tensor) = self.tensor_graph.get_tensor(tensor_id.0, &tensor_id.1) {
            tensor.data.location_string()
        } else {
            "Unknown".to_string()
        }
    }

    pub fn calculate_layer_parameters(&self, layer_id: LayerId) -> usize {
        if let Some(layer) = self.model.layers.get(&layer_id) {
            let input_tensor_ids = self
                .tensor_graph
                .get_layer_tensor_ids(layer_id)
                .into_iter()
                .filter(|id| id.1.starts_with("input"))
                .collect::<Vec<_>>();

            let input_shapes: Vec<&TensorDesc> = input_tensor_ids
                .iter()
                .filter_map(|id| self.tensor_graph.get_tensor(id.0, &id.1))
                .map(|tensor| &tensor.desc)
                .collect();

            return layer
                .layer
                .parameter_count(self.model.batch_size, &input_shapes);
        }
        0
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
            self.tensor_graph.tensors = HashMap::new();
        }
        {
            self.gpus = Vec::new();
        }
    }
}
