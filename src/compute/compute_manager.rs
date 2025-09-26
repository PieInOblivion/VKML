use std::collections::HashMap;
use std::ptr;
use std::sync::Arc;

use crate::compute::{print_model_stats, print_tensorgraph_stats};
use crate::gpu::vk_gpu::GpuPool;
use crate::importers::onnx_parser::OnnxParser;
use crate::instruction;
use crate::tensor::cell::TensorCell;
use crate::tensor::tensor::{DeviceId, Tensor};
use crate::utils::error::VKMLError;
use onnx_extractor::OnnxModel;
use zero_pool::{ZeroPool, zp_define_task_fn};

use crate::instruction::instruction::Instruction;
use crate::tensor_graph::tensor_graph::{OperationId, TensorGraph};
use crate::{
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
    pub tensors: Vec<TensorCell>,

    pub model: GraphModel,
    pub tensor_graph: TensorGraph,

    gpus: Arc<GpuPool>,
    cpu: CPUCompute,
    thread_pool: Arc<ZeroPool>,
}

impl ComputeManager {
    pub fn new(model: GraphModel, thread_pool: Arc<ZeroPool>) -> Result<Self, VKMLError> {
        let gpus = GpuPool::new(None)?;
        Self::new_with(model, thread_pool, gpus, None)
    }

    pub fn new_with(
        mut model: GraphModel,
        thread_pool: Arc<ZeroPool>,
        gpus: GpuPool,
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
            tensor_graph: tensor_graph,
            gpus: Arc::new(gpus),
            cpu,
            thread_pool,
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

    pub fn new_onnx(onnx_path: &str, thread_pool: Arc<ZeroPool>) -> Result<Self, VKMLError> {
        let onnx_model = OnnxModel::load_from_file(onnx_path).map_err(|e| {
            VKMLError::OnnxImporterError(format!(
                "Failed to load ONNX model from '{}': {}",
                onnx_path, e
            ))
        })?;

        let (tensor_graph, tensor_bytes) = OnnxParser::parse_onnx_model(onnx_model)?;

        let gpus = GpuPool::new(None)?;
        Self::new_from_tensor_graph_with(tensor_graph, tensor_bytes, thread_pool, gpus, None)
    }

    /// Create ComputeManager from ONNX file with custom settings
    pub fn new_onnx_with(
        onnx_path: &str,
        thread_pool: Arc<ZeroPool>,
        gpus: GpuPool,
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
        gpus: GpuPool,
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
        // Get execution plan and flatten to a linear sequence of operations
        let execution_plan = self.tensor_graph.create_execution_plan();
        let flattened_ops: Vec<OperationId> = execution_plan.into_iter().flatten().collect();

        // Track planned tensor locations: tensor_id -> DeviceLocation
        let mut tensor_locations: Vec<Option<DeviceLocation>> =
            vec![None; self.tensor_graph.tensor_descs.len()];

        // Maintain a list of tensor remappings: (original_id, device) -> new_id
        let mut tensor_remappings: HashMap<(usize, DeviceLocation), usize> = HashMap::new();

        // Store remappings needed for operations: op_id -> (new_inputs, new_outputs)
        let mut operation_remappings: HashMap<OperationId, (Vec<usize>, Vec<usize>)> =
            HashMap::new();

        // New tensors created for transfers or device-local outputs - including layer info
        let mut new_tensors: Vec<(TensorDesc, DeviceLocation, Option<LayerId>)> = Vec::new();

        // Transfer operations to insert: (insert_before_op, transfer_instr)
        let mut transfer_operations: Vec<(OperationId, Box<dyn Instruction>)> = Vec::new();

        // Track available memory per device in the desired order (GPUs then CPU)
        let mut available_memory: Vec<(DeviceLocation, u64)> = Vec::new();
        for (idx, gpu) in self.gpus.gpus().iter().enumerate() {
            available_memory.push((DeviceLocation::GPU(idx), gpu.available_memory()));
        }
        available_memory.push((
            DeviceLocation::CPU,
            self.cpu.memory_tracking.get_available(),
        ));

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
                            if !tensor_remappings.contains_key(&(tid, cand_device.clone())) {
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
                            if !tensor_remappings.contains_key(&(tid, cand_device.clone())) {
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
                        let key = (tid, current_device.clone());
                        if let Some(&mapped_id) = tensor_remappings.get(&key) {
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

                            tensor_remappings.insert(key, new_tensor_id);
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
                        let key = (tid, current_device.clone());
                        if let Some(&mapped_id) = tensor_remappings.get(&key) {
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
                            tensor_remappings.insert(key, new_tensor_id);
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
                operation_remappings.insert(op_id, (new_inputs, new_outputs));
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
        let mut remap_entries: Vec<(OperationId, Vec<usize>, Vec<usize>)> = operation_remappings
            .into_iter()
            .map(|(op, (ins, outs))| (op, ins, outs))
            .collect();
        remap_entries.sort_by_key(|e| e.0);

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
        tensor_locations: Vec<Option<DeviceLocation>>,
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

        self.thread_pool
            .submit_batch_uniform(single_allocate_task, &tasks)
            .wait();

        unsafe { self.tensors.set_len(count) };

        Ok(())
    }

    pub fn allocate_tensor(
        &self,
        desc: &TensorDesc,
        target_device: &DeviceLocation,
        init_box: Option<Box<[u8]>>,
    ) -> Result<Tensor, VKMLError> {
        let size_in_bytes = desc.size_in_bytes() as u64;

        match target_device {
            DeviceLocation::CPU => {
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
            DeviceLocation::GPU(idx) => {
                let gpu_idx = *idx;
                let gpu = &self.gpus.get_gpu(gpu_idx);

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
                    Ok(Tensor::new_gpu(desc.clone(), gpu_idx, gpu_mem))
                } else {
                    let gpu_mem = gpu
                        .allocate_uninitialised_gpu_memory(size_in_bytes as usize)
                        .map_err(|e| VKMLError::VulkanError(e.to_string()))?;
                    Ok(Tensor::new_gpu(desc.clone(), gpu_idx, gpu_mem))
                }
            }
        }
    }

    pub fn forward(&mut self, batches: Vec<Tensor>) -> Result<Vec<Tensor>, VKMLError> {
        // Get input tensor indices
        let input_tensor_ids = self.tensor_graph.get_input_tensor_ids();

        // Validate input batch count
        if batches.len() != input_tensor_ids.len() {
            return Err(VKMLError::VulkanError(format!(
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
                return Err(VKMLError::VulkanError(format!(
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

        Err(VKMLError::VulkanError(format!(
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

    pub fn tensor_write(&self, tensor_id: usize) -> &mut Tensor {
        unsafe { self.tensors[tensor_id].as_mut() }
    }
}

struct SingleAllocParams {
    index: usize,
    initialisers_ptr: *mut Option<Box<[u8]>>,
    manager_ptr: *const ComputeManager,
    out_ptrs: *mut TensorCell,
    tensor_locations_ptr: *const Option<DeviceLocation>,
}

zp_define_task_fn!(single_allocate_task, SingleAllocParams, |params| {
    let init_box = unsafe { (*params.initialisers_ptr.add(params.index)).take() };

    let manager: &ComputeManager = unsafe { &*params.manager_ptr };

    let desc: &TensorDesc = &manager.tensor_graph.tensor_descs[params.index];

    let target = unsafe {
        (*params.tensor_locations_ptr.add(params.index))
            .clone()
            .unwrap_or(DeviceLocation::CPU)
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
        let gpu = params.compute_manager.gpus.get_gpu(params.gpu_idx);

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
