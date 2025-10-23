use crate::{
    instruction::Instruction,
    layer::execution::LayerExecution,
    model::{graph_model::GraphModel, layer_connection::LayerId},
    tensor::TensorDesc,
    utils::error::VKMLError,
};
use std::collections::{HashMap, HashSet, VecDeque};

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

/// Cached dependency graph information for operations.
/// Built once and reused by allocation and execution planning.
pub struct DependencyGraph {
    pub predecessors: Vec<Vec<OperationId>>,
    pub successors: Vec<Vec<OperationId>>,
    pub topological_order: Vec<OperationId>,
}

pub struct TensorGraph {
    pub tensor_descs: Vec<TensorDesc>,
    pub operations: Vec<Box<dyn Instruction>>,

    // Graph entry and exit points (indices into tensor_descs)
    pub input_tensor_ids: Vec<TensorId>,
    pub output_tensor_ids: Vec<TensorId>,

    // Vector mapping from tensor indices to layer IDs
    pub tensor_to_layer: Vec<Option<LayerId>>,
    pub operation_to_layer: Vec<LayerId>,

    pub memory_requirements: usize,
}

impl TensorGraph {
    pub fn from_graph_model(model: &GraphModel) -> Result<Self, VKMLError> {
        if model.verified.is_none() {
            return Err(VKMLError::TensorGraph("Model not verified".into()));
        }

        let execution_order = &model.verified.as_ref().unwrap().execution_order;
        let mut tensor_descs: Vec<TensorDesc> = Vec::new();
        let mut operations: Vec<Box<dyn Instruction>> = Vec::new();
        let mut tensor_to_layer_map = Vec::new();
        let mut operation_to_layer_map = Vec::new();

        let mut global_tensor_map: HashMap<(LayerId, usize), TensorId> = HashMap::new();
        let mut layer_executions: HashMap<LayerId, LayerExecution> = HashMap::new();

        let mut memory_requirements = 0;

        // --- Pass 1: Build LayerExecutions (determines local tensor descs and ops for each layer) ---
        for &layer_id in execution_order {
            let layer_wrapper = model.layers.get(&layer_id).ok_or_else(|| {
                VKMLError::TensorGraph(format!("Layer {} not found in model", layer_id))
            })?;

            let input_descs: Vec<TensorDesc> = layer_wrapper
                .input_connections
                .iter()
                .map(|conn| {
                    let src_layer_id = conn.get_layerid();
                    let src_output_idx = conn.get_outputidx();
                    let src_exec = layer_executions.get(&src_layer_id).ok_or_else(|| {
                        VKMLError::TensorGraph(format!(
                            // Changed to InternalError
                            "Source LayerExecution for {} not found when building layer {}",
                            src_layer_id, layer_id
                        ))
                    })?;
                    // Get the local tensor index within the source layer's execution
                    let src_local_tensor_idx = src_exec.outputs[src_output_idx];
                    Ok(src_exec.tensors[src_local_tensor_idx].clone())
                })
                .collect::<Result<Vec<TensorDesc>, VKMLError>>()?;

            let input_desc_refs: Vec<&TensorDesc> = input_descs.iter().collect();
            let layer_exec = layer_wrapper
                .layer
                .build_layer_exec(model.batch_size, &input_desc_refs)?;
            layer_executions.insert(layer_id, layer_exec);
        }

        // --- Pass 2: Create Global Tensors and Operations, and map local to global ---
        // `latest_producer_op_for_tensor[global_tensor_id]` stores the OperationId that last wrote to this tensor.
        let mut latest_producer_op_for_tensor: Vec<Option<OperationId>> = Vec::new();

        for &layer_id in execution_order {
            // Process layers in their execution order
            // First do read-only work while borrowing immutably in a short scope.
            {
                let layer_exec = layer_executions.get(&layer_id).unwrap();

                // Create global tensors for this layer's *newly defined* local tensors
                for (local_idx, local_tensor_desc) in layer_exec.tensors.iter().enumerate() {
                    if !layer_exec.input_mappings.contains_key(&local_idx) {
                        // Only if not an input reference
                        let global_tensor_id = tensor_descs.len();
                        global_tensor_map.insert((layer_id, local_idx), global_tensor_id);
                        memory_requirements += local_tensor_desc.size_in_bytes();
                        tensor_descs.push(local_tensor_desc.clone());
                        tensor_to_layer_map.push(Some(layer_id));
                        // Ensure latest_producer_op_for_tensor is large enough
                        if global_tensor_id >= latest_producer_op_for_tensor.len() {
                            latest_producer_op_for_tensor.resize(global_tensor_id + 1, None);
                        }
                        // Initially, newly defined tensors don't have a producer op from *within this graph's ops*
                        // unless they are model inputs (handled later) or produced by an op in this layer.
                    }
                }
                // Map input references to their global IDs (already created by producer layers)
                for (local_idx, (input_conn_idx, _output_idx_in_conn)) in &layer_exec.input_mappings
                {
                    let input_connection =
                        &model.layers[&layer_id].input_connections[*input_conn_idx];
                    let src_layer_id = input_connection.get_layerid();
                    let src_local_output_idx =
                        layer_executions[&src_layer_id].outputs[input_connection.get_outputidx()];
                    let global_src_tensor_id =
                        global_tensor_map[&(src_layer_id, src_local_output_idx)];
                    global_tensor_map.insert((layer_id, *local_idx), global_src_tensor_id);
                }
            }

            // Now that the immutable borrow ended, take mutable borrow to move instructions
            // Create operations for this layer (move instructions out of the layer)
            let layer_exec_mut = layer_executions.get_mut(&layer_id).unwrap();
            for mut instruction in layer_exec_mut.instructions.drain(..) {
                let global_op_id = operations.len();

                let global_inputs: Vec<TensorId> = instruction
                    .get_input_tensor_ids()
                    .iter()
                    .map(|&local_id| global_tensor_map[&(layer_id, local_id)])
                    .collect();
                let global_outputs: Vec<TensorId> = instruction
                    .get_output_tensor_ids()
                    .iter()
                    .map(|&local_id| global_tensor_map[&(layer_id, local_id)])
                    .collect();

                instruction.remap_tensor_ids(&global_inputs, &global_outputs);
                operations.push(instruction);
                operation_to_layer_map.push(layer_id);

                // Update latest_producer_op_for_tensor for all outputs of this new global_op_id.
                // This correctly handles in-place: this op is now the latest writer.
                for &output_global_id in &global_outputs {
                    if output_global_id >= latest_producer_op_for_tensor.len() {
                        // Should be covered by earlier resize
                        latest_producer_op_for_tensor.resize(output_global_id + 1, None);
                    }
                    latest_producer_op_for_tensor[output_global_id] = Some(global_op_id);
                }
            }
        }

        // --- Pass 4: Identify Model Input and Output Tensors ---
        // Entry tensors = output tensors of the model's entry‐point layers
        let mut input_tensors_model = Vec::new();
        for &layer_id in &model.verified.as_ref().unwrap().entry_points {
            let layer_exec = layer_executions.get(&layer_id).unwrap();
            for (local_idx, _) in layer_exec.tensors.iter().enumerate() {
                // only newly defined tensors (not input refs)
                if !layer_exec.input_mappings.contains_key(&local_idx) {
                    let global_id = global_tensor_map[&(layer_id, local_idx)];
                    input_tensors_model.push(global_id);
                }
            }
        }
        input_tensors_model.sort_unstable();

        let mut output_tensors_model = Vec::new();
        // Model outputs are typically the outputs of layers designated as exit points.
        // Or, more generally, tensors that are produced but not consumed by any other op in the graph.
        // For now, using the exit_points from GraphModel.
        let mut seen_outputs = HashSet::new();
        for &layer_id in &model.verified.as_ref().unwrap().exit_points {
            let layer_exec = layer_executions.get(&layer_id).unwrap();
            for &local_output_idx in &layer_exec.outputs {
                if let Some(global_id) = global_tensor_map.get(&(layer_id, local_output_idx))
                    && seen_outputs.insert(*global_id)
                {
                    output_tensors_model.push(*global_id);
                }
            }
        }
        output_tensors_model.sort_unstable();

        Ok(TensorGraph {
            tensor_descs,
            operations,
            input_tensor_ids: input_tensors_model,
            output_tensor_ids: output_tensors_model,
            tensor_to_layer: tensor_to_layer_map,
            operation_to_layer: operation_to_layer_map,
            memory_requirements,
        })
    }

    pub fn dependency_graph(&self) -> DependencyGraph {
        let num_ops = self.operations.len();
        let mut predecessors: Vec<Vec<OperationId>> = vec![Vec::new(); num_ops];
        let mut successors: Vec<Vec<OperationId>> = vec![Vec::new(); num_ops];

        for (curr_op, op) in self.operations.iter().enumerate().take(num_ops) {
            let mut preds = HashSet::new();
            for &t in &op.get_input_tensor_ids() {
                for pred_op in self.get_tensor_producers(t) {
                    if pred_op != curr_op && preds.insert(pred_op) {
                        predecessors[curr_op].push(pred_op);
                        successors[pred_op].push(curr_op);
                    }
                }
            }
        }

        // compute topological order using kahns algorithm
        let mut in_degree: Vec<usize> = predecessors.iter().map(|p| p.len()).collect();
        let mut dq: VecDeque<OperationId> = VecDeque::new();
        for (op, &deg) in in_degree.iter().enumerate().take(num_ops) {
            if deg == 0 {
                dq.push_back(op);
            }
        }

        let mut ordered: Vec<OperationId> = Vec::with_capacity(num_ops);
        let mut scheduled = 0;

        while let Some(op) = dq.pop_front() {
            ordered.push(op);
            scheduled += 1;
            for &succ in &successors[op] {
                in_degree[succ] = in_degree[succ].saturating_sub(1);
                if in_degree[succ] == 0 {
                    dq.push_back(succ);
                }
            }
        }

        if scheduled < num_ops {
            eprintln!(
                "Could not schedule all operations: {}/{}.",
                scheduled, num_ops
            );
        }

        DependencyGraph {
            predecessors,
            successors,
            topological_order: ordered,
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

    pub fn get_input_tensor_ids(&self) -> &[TensorId] {
        &self.input_tensor_ids
    }

    pub fn get_output_tensor_ids(&self) -> &[TensorId] {
        &self.output_tensor_ids
    }
}
