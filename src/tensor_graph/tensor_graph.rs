use crate::{
    dataloader::error::VKMLEngineError,
    gpu::gpu_memory::GPUMemory,
    instruction::instruction::Instruction,
    layer::execution::LayerExecution,
    model::{graph_model::GraphModel, layer_connection::LayerId},
    tensor::{compute_tensor::ComputeTensor, tensor_data::TensorData, tensor_desc::TensorDesc},
};
use std::collections::{HashMap, HashSet};

// TODO:
// This representation of tensor dag needs changing.
// Currently it stores layer information as an easy way to transition the layer graph into a tensor graph
// But the human readability should be able to be added a more effecient way
// I've thought about a universal representation, instead of the graph to tensor conversions,
// But the layer graph doesn't add that much over head, and it's pretty intuitive to use and edit new layers with, for users and me.
// Currently we will stick with the two forms of representation.

// Unique identifier for a tensor operation
pub type OperationId = usize;

// Unique identifier for a tensor
pub type TensorId = usize;

pub struct TensorGraph {
    pub tensors: Vec<ComputeTensor>, // Indexed by global TensorId
    pub operations: Vec<Box<dyn Instruction>>, // Using global TensorIds

    // Graph entry and exit points
    pub input_tensors: Vec<TensorId>,
    pub output_tensors: Vec<TensorId>,

    // Vector mapping from tensor indices to layer IDs
    pub tensor_to_layer: Vec<Option<LayerId>>,
}

impl TensorGraph {
    pub fn from_graph_model(model: &GraphModel) -> Result<Self, VKMLEngineError> {
        if model.verified.is_none() {
            return Err(VKMLEngineError::VulkanLoadError(
                "Model not verified".into(),
            ));
        }

        let execution_order = &model.verified.as_ref().unwrap().execution_order;
        let mut tensors = Vec::new();
        let mut operations = Vec::new();
        let mut input_tensors = Vec::new();
        let mut output_tensors = Vec::new();

        // Map from (layer_id, local_tensor_idx) to global tensor index
        let mut tensor_mapping = HashMap::new();

        let mut tensor_to_layer = Vec::new();

        // First pass: Build layer executions
        let mut layer_executions: HashMap<usize, LayerExecution> = HashMap::new();
        for &layer_id in execution_order {
            let layer = model.layers.get(&layer_id).ok_or_else(|| {
                VKMLEngineError::VulkanLoadError(format!("Layer {} not found in model", layer_id))
            })?;

            // Get input shapes
            let input_shapes: Vec<TensorDesc> = layer
                .input_connections
                .iter()
                .map(|connection| {
                    let input_id = connection.get_layerid();
                    let output_idx = connection.get_outputidx();

                    if let Some(exec) = layer_executions.get(&input_id) {
                        if output_idx < exec.outputs.len() {
                            let output_tensor_idx = exec.outputs[output_idx];
                            return Ok(exec.tensors[output_tensor_idx].clone());
                        }
                    }

                    Err(VKMLEngineError::VulkanLoadError(format!(
                        "Could not find output tensor for layer {} at index {}",
                        input_id, output_idx
                    )))
                })
                .collect::<Result<Vec<TensorDesc>, VKMLEngineError>>()?;

            let input_shape_refs: Vec<&TensorDesc> = input_shapes.iter().collect();

            // Build layer execution
            let layer_exec = layer
                .layer
                .build_layer_exec(model.batch_size, &input_shape_refs)?;

            // Store layer execution for later use
            layer_executions.insert(layer_id, layer_exec);
        }

        // Second pass: Process non-input tensors and create global array
        for &layer_id in execution_order {
            let layer_exec = layer_executions.get(&layer_id).unwrap();

            // Get the set of local tensor indices that are inputs
            let input_tensor_indices: HashSet<TensorId> =
                layer_exec.input_mappings.keys().cloned().collect();

            // Process each tensor that isn't an input reference
            for local_idx in 0..layer_exec.tensors.len() {
                if !input_tensor_indices.contains(&local_idx) {
                    let global_idx = tensors.len();
                    tensor_mapping.insert((layer_id, local_idx), global_idx);
                    tensors.push(ComputeTensor {
                        desc: layer_exec.tensors[local_idx].clone(),
                        data: TensorData::Unallocated,
                    });
                    tensor_to_layer.push(Some(layer_id));
                }
            }
        }

        // Third pass: Process input mappings to connect tensors across layers
        for &layer_id in execution_order {
            let layer = model.layers.get(&layer_id).unwrap();
            let layer_exec = layer_executions.get(&layer_id).unwrap();

            for (local_idx, (input_idx, output_idx)) in &layer_exec.input_mappings {
                if *input_idx >= layer.input_connections.len() {
                    return Err(VKMLEngineError::VulkanLoadError(format!(
                        "Invalid input index {} in layer {}, only has {} inputs",
                        input_idx,
                        layer_id,
                        layer.input_connections.len()
                    )));
                }

                // Find the source layer and tensor
                let connection = &layer.input_connections[*input_idx];
                let source_layer_id = connection.get_layerid();
                let source_output_idx = connection.get_outputidx();

                let source_exec = layer_executions.get(&source_layer_id).unwrap();
                if source_output_idx >= source_exec.outputs.len() {
                    return Err(VKMLEngineError::VulkanLoadError(format!(
                        "Invalid output index {} in layer {}, only has {} outputs",
                        source_output_idx,
                        source_layer_id,
                        source_exec.outputs.len()
                    )));
                }

                let source_local_idx = source_exec.outputs[source_output_idx];

                // Map this tensor to the global index of its source
                let source_global_idx = tensor_mapping[&(source_layer_id, source_local_idx)];
                tensor_mapping.insert((layer_id, *local_idx), source_global_idx);
            }
        }

        // Fourth pass: Process instructions
        for &layer_id in execution_order {
            let layer_exec = layer_executions.get(&layer_id).unwrap();

            for instruction in &layer_exec.instructions {
                // Get input and output tensor indices
                let local_inputs = instruction.get_input_tensor_ids();
                let local_outputs = instruction.get_output_tensor_ids();

                // Map to global indices
                let global_inputs: Vec<usize> = local_inputs
                    .iter()
                    .map(|&local_id| tensor_mapping[&(layer_id, local_id)])
                    .collect();

                let global_outputs: Vec<usize> = local_outputs
                    .iter()
                    .map(|&local_id| tensor_mapping[&(layer_id, local_id)])
                    .collect();

                // Create instruction with global indices
                let mut remapped = instruction.clone();
                remapped.remap_tensor_ids(&global_inputs, &global_outputs);
                operations.push(remapped);
            }
        }

        // Fifth pass: Identify model input and output tensors
        for &layer_id in &model.verified.as_ref().unwrap().entry_points {
            let layer_exec = layer_executions.get(&layer_id).unwrap();
            for &output_idx in &layer_exec.outputs {
                let global_idx = tensor_mapping[&(layer_id, output_idx)];
                input_tensors.push(global_idx);
            }
        }

        for &layer_id in &model.verified.as_ref().unwrap().exit_points {
            let layer_exec = layer_executions.get(&layer_id).unwrap();
            for &output_idx in &layer_exec.outputs {
                let global_idx = tensor_mapping[&(layer_id, output_idx)];
                output_tensors.push(global_idx);
            }
        }

        Ok(TensorGraph {
            tensors,
            operations,
            input_tensors,
            output_tensors,
            tensor_to_layer,
        })
    }

    pub fn create_execution_plan(&self) -> Vec<Vec<OperationId>> {
        let mut execution_plan = Vec::new();
        let mut pending_ops = Vec::new();
        let mut tensor_ready = vec![false; self.tensors.len()];

        // Build operation dependencies
        let mut operation_inputs: Vec<Vec<usize>> = Vec::with_capacity(self.operations.len());
        let mut operation_outputs: Vec<Vec<usize>> = Vec::with_capacity(self.operations.len());

        // Process operations to build dependency information
        for (op_idx, op) in self.operations.iter().enumerate() {
            // Track which tensors each operation reads
            let inputs = op.get_input_tensor_ids();
            operation_inputs.push(inputs.clone());

            // Track which tensors each operation writes
            let outputs = op.get_output_tensor_ids();
            operation_outputs.push(outputs.clone());

            // Initialize pending operations list
            pending_ops.push(op_idx);
        }

        // Mark input tensors as ready
        for &tensor_id in &self.input_tensors {
            tensor_ready[tensor_id] = true;
        }

        // Build parameter tensor set - these tensors have no producers
        let mut tensor_has_producer = vec![false; self.tensors.len()];
        for op in &self.operations {
            for &output_id in &op.get_output_tensor_ids() {
                tensor_has_producer[output_id] = true;
            }
        }

        // Mark parameter tensors as ready
        for (tensor_idx, has_producer) in tensor_has_producer.iter().enumerate() {
            if !has_producer && !self.input_tensors.contains(&tensor_idx) {
                tensor_ready[tensor_idx] = true;
            }
        }

        // Build execution plan until all operations are scheduled
        while !pending_ops.is_empty() {
            let mut ready_ops = Vec::new();
            let mut stage_read_tensors = HashSet::new(); // Tensors read in this stage
            let mut stage_write_tensors = HashSet::new(); // Tensors written in this stage

            // Find operations whose inputs are all ready and don't conflict with operations in this stage
            let mut i = 0;
            while i < pending_ops.len() {
                let op_idx = pending_ops[i];

                // Check if all inputs are ready
                let all_inputs_ready = operation_inputs[op_idx]
                    .iter()
                    .all(|&input_id| tensor_ready[input_id]);

                if all_inputs_ready {
                    // Check for conflicts with operations already in this stage
                    let op_inputs = &operation_inputs[op_idx];
                    let op_outputs = &operation_outputs[op_idx];

                    // Check for write-read conflicts:
                    // If this op reads a tensor that another op in this stage writes to
                    let read_conflict = op_inputs
                        .iter()
                        .any(|&input_id| stage_write_tensors.contains(&input_id));

                    // Check for write-write conflicts and read-write conflicts:
                    // If this op writes to a tensor that another op in this stage reads from or writes to
                    let write_conflict = op_outputs.iter().any(|&output_id| {
                        stage_read_tensors.contains(&output_id)
                            || stage_write_tensors.contains(&output_id)
                    });

                    if !read_conflict && !write_conflict {
                        // Add operation to this stage
                        ready_ops.push(op_idx);

                        // Track tensors read and written by this operation
                        for &input_id in op_inputs {
                            stage_read_tensors.insert(input_id);
                        }

                        for &output_id in op_outputs {
                            stage_write_tensors.insert(output_id);
                        }

                        // Remove the operation from pending list
                        pending_ops.swap_remove(i);
                    } else {
                        // This operation has conflicts, try it in the next stage
                        i += 1;
                    }
                } else {
                    // Inputs not ready, try next operation
                    i += 1;
                }
            }

            // Handle potential dependency issues (cycles, etc.)
            if ready_ops.is_empty() && !pending_ops.is_empty() {
                eprintln!(
                    "Warning: No operations ready but {} pending. Possible dependency issue.",
                    pending_ops.len()
                );
                // As a fallback, pick the first pending operation
                let op_idx = pending_ops.remove(0);
                ready_ops.push(op_idx);
            }

            // Add this stage to the execution plan
            if !ready_ops.is_empty() {
                // Convert to OperationId type for the final result
                let ready_op_ids: Vec<OperationId> = ready_ops.iter().map(|&idx| idx).collect();

                execution_plan.push(ready_op_ids);

                // Mark output tensors from these operations as ready
                for &op_idx in &ready_ops {
                    let outputs = &operation_outputs[op_idx];
                    for &output_id in outputs {
                        tensor_ready[output_id] = true;
                    }
                }
            }
        }

        execution_plan
    }

    pub fn get_gpu_memory_or_panic(&self, tensor_id: &TensorId) -> &GPUMemory {
        match &self.tensors[*tensor_id].data {
            TensorData::GPU { memory, .. } => memory,
            TensorData::CPU(_) => {
                panic!("Tensor {} is in CPU memory, expected GPU memory", tensor_id)
            }
            TensorData::Unallocated => {
                panic!("Tensor {} is unallocated, expected GPU memory", tensor_id)
            }
        }
    }

    pub fn get_instruction_or_panic(&self, idx: usize) -> &dyn Instruction {
        self.operations
            .get(idx)
            .map(|boxed| boxed.as_ref())
            .unwrap_or_else(|| panic!("Instruction index {} is out of bounds", idx))
    }

    // Get all operations that produce a given tensor
    pub fn get_tensor_producers(&self, tensor_id: usize) -> Vec<usize> {
        self.operations
            .iter()
            .enumerate()
            .filter_map(|(op_idx, op)| {
                if op.get_output_tensor_ids().contains(&tensor_id) {
                    Some(op_idx)
                } else {
                    None
                }
            })
            .collect()
    }

    // Get all operations that consume a given tensor
    pub fn get_tensor_consumers(&self, tensor_id: usize) -> Vec<usize> {
        self.operations
            .iter()
            .enumerate()
            .filter_map(|(op_idx, op)| {
                if op.get_input_tensor_ids().contains(&tensor_id) {
                    Some(op_idx)
                } else {
                    None
                }
            })
            .collect()
    }

    // Get all input tensors for a given operation
    pub fn get_operation_inputs(&self, op_idx: usize) -> Vec<usize> {
        if op_idx < self.operations.len() {
            self.operations[op_idx].get_input_tensor_ids()
        } else {
            Vec::new()
        }
    }

    // Get all output tensors for a given operation
    pub fn get_operation_outputs(&self, op_idx: usize) -> Vec<usize> {
        if op_idx < self.operations.len() {
            self.operations[op_idx].get_output_tensor_ids()
        } else {
            Vec::new()
        }
    }

    pub fn calculate_memory_requirements(&self) -> u64 {
        self.tensors
            .iter()
            .map(|tensor| tensor.desc.size_in_bytes() as u64)
            .sum()
    }

    pub fn get_dag_input_tensor_ids(&self) -> &[TensorId] {
        &self.input_tensors
    }

    pub fn get_dag_output_tensor_ids(&self) -> &[TensorId] {
        &self.output_tensors
    }
}
