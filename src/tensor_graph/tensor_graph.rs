use crate::{
    dataloader::error::VKMLEngineError,
    gpu::gpu_memory::GPUMemory,
    layer::execution::LayerExecution,
    model::{graph_model::GraphModel, instruction::Instruction, layer_connection::LayerId},
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
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct OperationId(pub LayerId, pub usize); // (layer_id, instruction_index)

// Unique identifier for a tensor
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TensorId(pub LayerId, pub String); // (layer_id, tensor_name)

pub struct TensorGraph {
    pub tensors: HashMap<TensorId, ComputeTensor>,
    pub operations: HashMap<OperationId, Instruction>,
    pub input_tensors: Vec<TensorId>,
    pub output_tensors: Vec<TensorId>,

    // Dependency tracking at tensor level
    pub tensor_dependencies: HashMap<TensorId, HashSet<OperationId>>, // Which operations produce this tensor
    pub operation_inputs: HashMap<OperationId, HashSet<TensorId>>, // Which tensors an operation reads
    pub operation_outputs: HashMap<OperationId, HashSet<TensorId>>, // Which tensors an operation writes
}

fn handle_circular_dependency(
    tensors: &mut HashMap<TensorId, ComputeTensor>,
    tensor_dependencies: &mut HashMap<TensorId, HashSet<OperationId>>,
    op_id: &OperationId,
    tensor_id: &TensorId,
    is_circular: bool,
) -> TensorId {
    if !is_circular {
        // No circular dependency, use original tensor
        if let Some(deps) = tensor_dependencies.get_mut(tensor_id) {
            deps.insert(op_id.clone());
        }
        return tensor_id.clone();
    }

    // Create temporary tensor to break cycle
    let temp_id = TensorId(tensor_id.0, format!("{}_temp", tensor_id.1));

    if let Some(original) = tensors.get(tensor_id) {
        // Clone the tensor with same descriptor but unallocated
        tensors.insert(
            temp_id.clone(),
            ComputeTensor {
                desc: original.desc.clone(),
                data: TensorData::Unallocated,
            },
        );

        // Create dependency entry for the temp tensor
        let mut deps = HashSet::new();
        deps.insert(op_id.clone());
        tensor_dependencies.insert(temp_id.clone(), deps);
    }

    temp_id
}

impl TensorGraph {
    pub fn from_graph_model(model: &GraphModel) -> Result<Self, VKMLEngineError> {
        if model.verified.is_none() {
            return Err(VKMLEngineError::VulkanLoadError(
                "Model not verified".into(),
            ));
        }

        let execution_order = &model.verified.as_ref().unwrap().execution_order;
        let mut tensors: HashMap<TensorId, ComputeTensor> = HashMap::new();
        let mut operations = HashMap::new();
        let mut input_tensors = Vec::new();
        let mut output_tensors = Vec::new();
        let mut tensor_dependencies = HashMap::new();
        let mut operation_inputs = HashMap::new();
        let mut operation_outputs = HashMap::new();
        let mut layer_id_map = HashMap::new();

        // Mapping from layer output to actual tensor ID
        let mut layer_output_mapping: HashMap<(LayerId, usize), TensorId> = HashMap::new();
        let mut layer_executions: HashMap<LayerId, LayerExecution> = HashMap::new();

        // Set to track parameter tensors (weights, biases)
        let mut parameter_tensors = HashSet::new();

        // PHASE 1: First build all tensors and their shapes
        for &layer_id in execution_order {
            let layer = model.layers.get(&layer_id).ok_or_else(|| {
                VKMLEngineError::VulkanLoadError(format!("Layer {} not found in model", layer_id))
            })?;

            // Get input shapes from connected layers
            let input_shapes: Vec<TensorDesc> = layer
                .input_connections
                .iter()
                .map(|connection| {
                    let input_id = connection.get_layerid();
                    let output_idx = connection.get_outputidx();

                    // Lookup the actual tensor from the layer's output
                    if let Some(exec) = layer_executions.get(&input_id) {
                        if output_idx < exec.outputs.len() {
                            let output_name = &exec.outputs[output_idx];
                            if let Some(tensor) =
                                tensors.get(&TensorId(input_id, output_name.clone()))
                            {
                                return Ok(tensor.desc.clone());
                            }
                        }
                    }

                    Err(VKMLEngineError::VulkanLoadError(format!(
                        "Could not find output tensor for layer {} at index {}",
                        input_id, output_idx
                    )))
                })
                .collect::<Result<Vec<TensorDesc>, VKMLEngineError>>()?;

            let input_shape_refs: Vec<&TensorDesc> = input_shapes.iter().collect();

            // Build layer execution using the correct input shapes
            let layer_exec = layer
                .layer
                .build_layer_exec(model.batch_size, &input_shape_refs)?;

            // Create tensors for this layer
            for (name, tensor_desc) in &layer_exec.tensors {
                let tensor_id = TensorId(layer_id, name.clone());

                let compute_tensor = ComputeTensor {
                    desc: tensor_desc.clone(),
                    data: TensorData::Unallocated,
                };

                tensors.insert(tensor_id.clone(), compute_tensor);
                tensor_dependencies.insert(tensor_id.clone(), HashSet::new());

                // Identify parameter tensors by checking if they have no dependencies
                // (rather than by name)
            }

            // Map layer outputs to actual tensor IDs
            for (idx, output_name) in layer_exec.outputs.iter().enumerate() {
                layer_output_mapping
                    .insert((layer_id, idx), TensorId(layer_id, output_name.clone()));
            }

            // Identify model inputs and outputs
            if layer.input_connections.is_empty() {
                for output_name in &layer_exec.outputs {
                    let tensor_id = TensorId(layer_id, output_name.clone());
                    input_tensors.push(tensor_id);
                }
            }

            if layer.output_connections.is_empty() {
                for output_name in &layer_exec.outputs {
                    let tensor_id = TensorId(layer_id, output_name.clone());
                    output_tensors.push(tensor_id);
                }
            }

            // Store the layer execution for phase 2
            layer_executions.insert(layer_id, layer_exec);
        }

        // PHASE 2: Create operations and establish dependencies
        for &layer_id in execution_order {
            let layer = model.layers.get(&layer_id).unwrap();
            let layer_exec = layer_executions.get(&layer_id).unwrap();

            let mut layer_ops = HashSet::new();

            // Process each instruction
            for (instr_idx, instruction) in layer_exec.instructions.iter().enumerate() {
                let op_id = OperationId(layer_id, instr_idx);
                operations.insert(op_id.clone(), instruction.clone());
                layer_ops.insert(op_id.clone());

                let mut op_inputs = HashSet::new();
                let mut op_outputs = HashSet::new();

                match instruction.get_all_input_tensor_ids(
                    layer_id,
                    &layer.input_connections,
                    &layer_executions,
                    &tensors,
                ) {
                    Ok(inputs) => op_inputs.extend(inputs),
                    Err(e) => {
                        println!(
                            "Warning: Failed to resolve inputs for operation {:?}: {}",
                            op_id, e
                        );
                    }
                }

                // Process all output tensors, handling circular dependencies
                let output_tensor_ids = instruction.get_output_tensor_ids(layer_id);

                for dst_tensor_id in output_tensor_ids {
                    //TODO: Is circular dep checking at the tensor level ever required?
                    // At the layer level already done, so it will have to be in-layer intentional
                    let output_id = handle_circular_dependency(
                        &mut tensors,
                        &mut tensor_dependencies,
                        &op_id,
                        &dst_tensor_id,
                        false,
                    );

                    op_outputs.insert(output_id);
                }

                // Store the operation's inputs and outputs
                operation_inputs.insert(op_id.clone(), op_inputs);
                operation_outputs.insert(op_id.clone(), op_outputs);
            }

            // Map layer to its operations
            layer_id_map.insert(layer_id, layer_ops);
        }

        // Identify parameter tensors by their dependency patterns
        for (tensor_id, deps) in &tensor_dependencies {
            // Parameters have no producers and are not inputs
            if deps.is_empty() && !input_tensors.contains(tensor_id) {
                parameter_tensors.insert(tensor_id.clone());
            }
        }

        // PHASE 3: Verify dependencies and fix any issues
        let mut missing_deps = 0;

        // Check each operation's inputs
        for (op_id, inputs) in &operation_inputs {
            for input in inputs {
                if !tensors.contains_key(input) {
                    println!(
                        "Warning: Operation {:?} depends on non-existent tensor {:?}",
                        op_id, input
                    );
                    missing_deps += 1;
                } else if !tensor_dependencies.contains_key(input)
                    || tensor_dependencies[input].is_empty()
                {
                    // Input exists but has no producers - this is fine for parameters and inputs
                    if !input_tensors.contains(input) && !parameter_tensors.contains(input) {
                        println!(
                            "Warning: Tensor {:?} has no producers but is not a model input or parameter",
                            input
                        );
                    }
                }
            }
        }

        if missing_deps > 0 {
            println!("Warning: {} missing dependencies found", missing_deps);
        }

        Ok(TensorGraph {
            tensors,
            operations,
            input_tensors,
            output_tensors,
            tensor_dependencies,
            operation_inputs,
            operation_outputs,
        })
    }

    pub fn create_execution_plan(&self) -> Vec<Vec<OperationId>> {
        // First, let's build a directed graph of operation dependencies
        let mut op_dependencies: HashMap<OperationId, HashSet<OperationId>> = HashMap::new();
        let mut op_dependents: HashMap<OperationId, HashSet<OperationId>> = HashMap::new();

        // Initialise with empty sets
        for op_id in self.operations.keys() {
            op_dependencies.insert(op_id.clone(), HashSet::new());
            op_dependents.insert(op_id.clone(), HashSet::new());
        }

        // First pass: Identify direct dependencies based on tensor flows
        for (op_id, inputs) in &self.operation_inputs {
            for input_tensor in inputs {
                // Find all operations that produce this input tensor
                if let Some(producers) = self.tensor_dependencies.get(input_tensor) {
                    // This operation depends on all producers of its input tensors
                    for producer in producers {
                        // Skip self-dependencies
                        if producer != op_id {
                            op_dependencies
                                .entry(op_id.clone())
                                .or_insert_with(HashSet::new)
                                .insert(producer.clone());

                            op_dependents
                                .entry(producer.clone())
                                .or_insert_with(HashSet::new)
                                .insert(op_id.clone());
                        }
                    }
                }
            }
        }

        // Second pass: Handle operations that modify tensors in-place
        // (same tensor as both input and output)
        let mut inplace_ops = HashSet::new();
        let mut tensor_modifiers: HashMap<TensorId, HashSet<OperationId>> = HashMap::new();

        for (op_id, outputs) in &self.operation_outputs {
            if let Some(inputs) = self.operation_inputs.get(op_id) {
                // Find tensors that are both input and output
                for tensor_id in inputs.intersection(outputs) {
                    inplace_ops.insert(op_id.clone());

                    // Record this operation as a modifier of this tensor
                    tensor_modifiers
                        .entry(tensor_id.clone())
                        .or_insert_with(HashSet::new)
                        .insert(op_id.clone());
                }
            }
        }

        // Third pass: Ensure operations that read a tensor depend on all operations
        // that modify it in-place
        for (tensor_id, modifiers) in &tensor_modifiers {
            // Find all operations that read this tensor
            for (op_id, inputs) in &self.operation_inputs {
                if inputs.contains(tensor_id) && !modifiers.contains(op_id) {
                    // This operation reads the tensor but doesn't modify it
                    for modifier in modifiers {
                        // Make the reader depend on the modifier
                        if modifier != op_id {
                            // Avoid self-dependencies
                            op_dependencies
                                .entry(op_id.clone())
                                .or_insert_with(HashSet::new)
                                .insert(modifier.clone());

                            op_dependents
                                .entry(modifier.clone())
                                .or_insert_with(HashSet::new)
                                .insert(op_id.clone());
                        }
                    }
                }
            }
        }

        // Topological sort, Kahn's algorithm
        let mut execution_plan = Vec::new();
        let mut remaining_ops: HashSet<_> = self.operations.keys().cloned().collect();
        let mut dependencies = op_dependencies.clone();

        // Continue until all operations are scheduled
        while !remaining_ops.is_empty() {
            // Find operations with no dependencies
            let mut ready_ops = Vec::new();

            for op_id in &remaining_ops {
                let deps = dependencies.get(op_id);
                if deps.is_none() || deps.unwrap().is_empty() {
                    ready_ops.push(op_id.clone());
                }
            }

            // If no operations are ready, we may have a cycle
            if ready_ops.is_empty() {
                // Find the operation in a cycle with the fewest dependencies
                let op_id = remaining_ops
                    .iter()
                    .min_by_key(|op_id| dependencies.get(*op_id).map_or(0, |deps| deps.len()))
                    .cloned()
                    .unwrap();

                ready_ops.push(op_id);

                // Log a cycle breaking event
                eprintln!(
                    "Warning: Breaking dependency cycle at operation {:?}",
                    ready_ops[0]
                );
            }

            // Add these operations to the current stage of the execution plan
            execution_plan.push(ready_ops.clone());

            // Remove these operations from the dependency graph
            for op_id in &ready_ops {
                // Remove this operation from remaining
                remaining_ops.remove(op_id);

                // Update dependencies for all operations that depend on this one
                if let Some(dependents) = op_dependents.get(op_id) {
                    for dependent in dependents {
                        if let Some(deps) = dependencies.get_mut(dependent) {
                            deps.remove(op_id);
                        }
                    }
                }
            }
        }

        execution_plan
    }

    pub fn get_tensor(&self, layer_id: LayerId, name: &str) -> Option<&ComputeTensor> {
        self.tensors.get(&TensorId(layer_id, name.to_string()))
    }

    pub fn get_tensor_mut(&mut self, layer_id: LayerId, name: &str) -> Option<&mut ComputeTensor> {
        self.tensors.get_mut(&TensorId(layer_id, name.to_string()))
    }

    pub fn calculate_memory_requirements(&self) -> u64 {
        self.tensors
            .values()
            .map(|tensor| tensor.desc.size_in_bytes() as u64)
            .sum()
    }

    pub fn calculate_layer_memory(&self, layer_id: LayerId) -> u64 {
        self.tensors
            .iter()
            .filter_map(|(id, tensor)| {
                if id.0 == layer_id {
                    Some(tensor.desc.size_in_bytes() as u64)
                } else {
                    None
                }
            })
            .sum()
    }

    pub fn get_layer_tensor_ids(&self, layer_id: LayerId) -> Vec<TensorId> {
        self.tensors
            .keys()
            .filter_map(|id| {
                if id.0 == layer_id {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_input_tensor_ids(&self) -> &[TensorId] {
        &self.input_tensors
    }

    pub fn get_output_tensor_ids(&self) -> &[TensorId] {
        &self.output_tensors
    }

    pub fn get_tensor_descriptor(&self, tensor_id: &TensorId) -> Option<&TensorDesc> {
        self.tensors.get(tensor_id).map(|tensor| &tensor.desc)
    }

    pub fn get_input_descriptors(&self) -> HashMap<TensorId, TensorDesc> {
        self.input_tensors
            .iter()
            .filter_map(|tensor_id| {
                self.get_tensor_descriptor(tensor_id)
                    .map(|desc| (tensor_id.clone(), desc.clone()))
            })
            .collect()
    }

    pub fn get_output_descriptors(&self) -> HashMap<TensorId, TensorDesc> {
        self.output_tensors
            .iter()
            .filter_map(|tensor_id| {
                self.get_tensor_descriptor(tensor_id)
                    .map(|desc| (tensor_id.clone(), desc.clone()))
            })
            .collect()
    }

    pub fn get_tensor_by_id_or_error(
        &self,
        id: &TensorId,
    ) -> Result<&ComputeTensor, VKMLEngineError> {
        self.tensors
            .get(id)
            .ok_or_else(|| VKMLEngineError::TensorNotFound(id.0, id.1.clone()))
    }

    pub fn get_operation_inputs_or_error(
        &self,
        op_id: &OperationId,
    ) -> Result<&HashSet<TensorId>, VKMLEngineError> {
        self.operation_inputs
            .get(op_id)
            .ok_or_else(|| VKMLEngineError::OperationNotFound(op_id.clone()))
    }

    pub fn get_operation_outputs_or_error(
        &self,
        op_id: &OperationId,
    ) -> Result<&HashSet<TensorId>, VKMLEngineError> {
        self.operation_outputs
            .get(op_id)
            .ok_or_else(|| VKMLEngineError::OperationNotFound(op_id.clone()))
    }

    pub fn get_gpu_memory_or_panic(&self, tensor_id: &TensorId) -> &GPUMemory {
        let tensor = self.get_tensor_by_id_or_error(tensor_id).unwrap();

        match &tensor.data {
            TensorData::GPU { memory, .. } => memory,
            _ => panic!("Tensor {:?} is not on GPU", tensor_id),
        }
    }
}
