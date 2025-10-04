use crate::{
    compute::compute_manager::DeviceLocation,
    instruction::instruction::Instruction,
    layer::execution::LayerExecution,
    model::{graph_model::GraphModel, layer_connection::LayerId},
    tensor::desc::TensorDesc,
    utils::error::VKMLError,
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

/// Represents a chunk of operations that can be submitted together to a device
#[derive(Debug, Clone)]
pub struct ExecutionChunk {
    pub device: DeviceLocation,
    pub operations: Vec<OperationId>,
    /// Dependencies within this chunk: (producer_op_index, consumer_op_index)
    /// Indices are into the operations vec above
    pub internal_deps: Vec<(usize, usize)>,
    /// Timeline semaphore values to wait on before executing: (device, value)
    pub wait_semaphores: Vec<(DeviceLocation, u64)>,
    /// Timeline semaphore value to signal when this chunk completes
    pub signal_value: u64,
    /// Whether this chunk requires host synchronization before execution
    pub requires_host_wait: bool,
}

/// Complete execution plan with dependency tracking for depth-aware scheduling
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub chunks: Vec<ExecutionChunk>,
    /// Next timeline semaphore value to use (monotonically increasing)
    pub next_semaphore_value: u64,
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
            return Err(VKMLError::VulkanError("Model not verified".into()));
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
                VKMLError::VulkanError(format!("Layer {} not found in model", layer_id))
            })?;

            let input_descs: Vec<TensorDesc> = layer_wrapper
                .input_connections
                .iter()
                .map(|conn| {
                    let src_layer_id = conn.get_layerid();
                    let src_output_idx = conn.get_outputidx();
                    let src_exec = layer_executions.get(&src_layer_id).ok_or_else(|| {
                        VKMLError::VulkanError(format!(
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
            let layer_exec = layer_executions.get(&layer_id).unwrap();

            // Create global tensors for this layer's *newly defined* local tensors
            for (local_idx, local_tensor_desc) in layer_exec.tensors.iter().enumerate() {
                if layer_exec.input_mappings.get(&local_idx).is_none() {
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
            for (local_idx, (input_conn_idx, _output_idx_in_conn)) in &layer_exec.input_mappings {
                let input_connection = &model.layers[&layer_id].input_connections[*input_conn_idx];
                let src_layer_id = input_connection.get_layerid();
                let src_local_output_idx =
                    layer_executions[&src_layer_id].outputs[input_connection.get_outputidx()];
                let global_src_tensor_id = global_tensor_map[&(src_layer_id, src_local_output_idx)];
                global_tensor_map.insert((layer_id, *local_idx), global_src_tensor_id);
            }

            // Create global operations for this layer
            for local_instruction in &layer_exec.instructions {
                let global_op_id = operations.len();
                let mut global_instruction = local_instruction.clone();

                let global_inputs: Vec<TensorId> = local_instruction
                    .get_input_tensor_ids()
                    .iter()
                    .map(|&local_id| global_tensor_map[&(layer_id, local_id)])
                    .collect();
                let global_outputs: Vec<TensorId> = local_instruction
                    .get_output_tensor_ids()
                    .iter()
                    .map(|&local_id| global_tensor_map[&(layer_id, local_id)])
                    .collect();

                global_instruction.remap_tensor_ids(&global_inputs, &global_outputs);
                operations.push(global_instruction);
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
                if layer_exec.input_mappings.get(&local_idx).is_none() {
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

    /// Creates a depth-aware execution plan for Vulkan 1.3+ using synchronization2 and timeline semaphores.
    ///
    /// # Requirements & Design
    ///
    /// ## Core Concept
    /// - **Depth-aware scheduling**: Track operation dependencies to submit chains of work in a single batch
    /// - **Minimize host synchronization**: Submit all work a device can do before needing to wait
    /// - **Timeline semaphores**: Coordinate multi-device execution without host intervention where possible
    /// - **Synchronization2**: Use VkDependencyInfo for fine-grained operation dependencies within submissions
    ///
    /// ## Execution Model
    ///
    /// ### Per-Device Submission Strategy
    /// - Group operations into **dependency chains** per device
    /// - Each chain is a sequence of operations where each depends on the previous
    /// - Independent chains on the same device can be submitted separately (round-robin across queues)
    /// - Submit entire chains with internal dependencies via synchronization2 in one `vkQueueSubmit2`
    ///
    /// ### Cross-Device Synchronization
    /// - **GPU-to-GPU transfers**: Use transfer operations (already in graph) + timeline semaphores
    ///   - Note: No direct GPU-to-GPU transfer for different vendors (must go through host)
    /// - **GPU-to-CPU or CPU-to-GPU**: Explicit host synchronization required
    /// - **Timeline semaphore values**: Track completion state per stage/submission, not per operation
    ///
    /// ### Host Synchronization Points
    /// Host must wait/signal when:
    /// 1. A CPU operation needs to execute (requires GPU results or produces GPU inputs)
    /// 2. Transfer operation requires host-side copy (different vendor GPUs)
    /// 3. End of graph execution (final outputs ready)
    ///
    /// ## Output Structure
    ///
    /// The execution plan should return information sufficient to:
    /// - Identify **submission chunks**: Groups of operations submitted together per device
    /// - Know **dependencies within chunks**: Which operations depend on which (for synchronization2)
    /// - Track **cross-chunk dependencies**: When one chunk depends on another (timeline semaphores)
    /// - Determine **host sync points**: Where CPU must wait before continuing
    ///
    /// Potential structure (to be designed):
    /// ```
    /// struct ExecutionChunk {
    ///     device: DeviceLocation,
    ///     operations: Vec<OperationId>,
    ///     internal_deps: Vec<(OperationId, OperationId)>, // (producer, consumer) pairs
    ///     wait_semaphores: Vec<(DeviceLocation, u64)>,    // (device, timeline_value) to wait on
    ///     signal_semaphore_value: u64,                     // value to signal when done
    /// }
    /// ```
    ///
    /// ## Allocation Independence
    /// - Tensor allocation strategy (`allocate_tensor_graph`) determines device placement
    /// - Execution plan works with whatever allocation decided
    /// - Graph may have operations on: GPU0 → GPU0 → CPU → GPU0 → GPU1, etc.
    ///
    /// ## Implementation Notes for execute()
    /// - Create timeline semaphores per device (persistent across execution)
    /// - Use `VkCommandBufferSubmitInfo` + `VkSubmitInfo2` for synchronization2
    /// - Each operation's command buffer gets dependency info via `VkDependencyInfo`
    /// - Track semaphore values to coordinate between chunks
    /// - Host waits only at required synchronization points (not after every stage)
    ///
    /// ## Changes Required
    /// - **This function** (`create_execution_plan`): Complete rewrite for depth-aware planning
    /// - **`execute()` in compute_manager.rs**: Rewrite to use synchronization2 + timeline semaphores
    /// - **`vk_gpu.rs`**: Update submission logic to use `vkQueueSubmit2` with dependency chains
    /// - **Extensions**: Synchronization2 and timeline_semaphore are core in Vulkan 1.3
    pub fn create_execution_plan(&self) -> Vec<Vec<OperationId>> {
        // Temporary: return old implementation until we have device info
        // This will be replaced once we refactor to use ExecutionPlan
        use std::collections::{HashSet, VecDeque};
        let num_ops = self.operations.len();
        let mut successors: Vec<Vec<OperationId>> = vec![Vec::new(); num_ops];
        let mut in_degree: Vec<usize> = vec![0; num_ops];

        for curr_op in 0..num_ops {
            let mut preds = HashSet::new();
            for &t in &self.operations[curr_op].get_input_tensor_ids() {
                for pred_op in self.get_tensor_producers(t) {
                    if pred_op != curr_op && preds.insert(pred_op) {
                        successors[pred_op].push(curr_op);
                    }
                }
            }
            in_degree[curr_op] = preds.len();
        }

        let mut plan = Vec::new();
        let mut dq: VecDeque<OperationId> = VecDeque::new();
        for op in 0..num_ops {
            if in_degree[op] == 0 {
                dq.push_back(op);
            }
        }
        let mut scheduled = 0;
        while scheduled < num_ops {
            if dq.is_empty() {
                eprintln!(
                    "Execution plan stuck: {}/{} scheduled. In-degrees: {:?}",
                    scheduled, num_ops, in_degree
                );
                break;
            }
            let mut stage = Vec::new();
            for _ in 0..dq.len() {
                let op = dq.pop_front().unwrap();
                stage.push(op);
                scheduled += 1;
                for &succ in &successors[op] {
                    in_degree[succ] -= 1;
                    if in_degree[succ] == 0 {
                        dq.push_back(succ);
                    }
                }
            }
            stage.sort_unstable();
            plan.push(stage);
        }
        if scheduled < num_ops {
            eprintln!(
                "Could not schedule all operations: {}/{}.",
                scheduled, num_ops
            );
        }
        plan
    }

    /// Creates depth-aware execution plan with proper device and dependency tracking
    pub fn create_execution_plan_v2(
        &self,
        device_locations: &[Option<DeviceLocation>],
    ) -> ExecutionPlan {
        use std::collections::HashMap;

        let num_ops = self.operations.len();

        // Pre-build tensor producer map for O(1) lookups instead of O(n) searches
        let mut tensor_producers: HashMap<TensorId, OperationId> = HashMap::new();
        for (op_idx, op) in self.operations.iter().enumerate() {
            for &tid in &op.get_output_tensor_ids() {
                tensor_producers.insert(tid, op_idx);
            }
        }

        // Build dependency graph with pre-sized vectors
        let mut successors: Vec<Vec<OperationId>> = vec![Vec::new(); num_ops];
        let mut predecessors: Vec<Vec<OperationId>> = vec![Vec::new(); num_ops];
        let mut in_degree: Vec<usize> = vec![0; num_ops];

        for curr_op in 0..num_ops {
            let input_ids = self.operations[curr_op].get_input_tensor_ids();
            let mut pred_set = HashSet::with_capacity(input_ids.len());

            for &t in &input_ids {
                if let Some(&pred_op) = tensor_producers.get(&t) {
                    if pred_op != curr_op && pred_set.insert(pred_op) {
                        successors[pred_op].push(curr_op);
                        predecessors[curr_op].push(pred_op);
                    }
                }
            }
            in_degree[curr_op] = pred_set.len();
        }

        // Determine device for each operation (single pass)
        let op_devices: Vec<DeviceLocation> = (0..num_ops)
            .map(|op_id| self.determine_op_device(op_id, device_locations))
            .collect();

        // Build chunks: collect all operations per device that can execute together
        let mut chunks = Vec::new();
        let mut scheduled = vec![false; num_ops];
        let mut scheduled_count = 0usize;
        let mut current_semaphore_value = 1u64;

        // Track the last semaphore value signaled by each device
        let mut device_semaphore_values: HashMap<DeviceLocation, u64> = HashMap::new();

        // Track current in-degrees (mutable copy)
        let mut current_in_degree = in_degree;

        while scheduled_count < num_ops {
            // Find all operations ready to execute (in-degree == 0) - optimized iteration
            let mut device_ops: HashMap<DeviceLocation, Vec<OperationId>> = HashMap::new();

            for op in 0..num_ops {
                if !scheduled[op] && current_in_degree[op] == 0 {
                    device_ops
                        .entry(op_devices[op].clone())
                        .or_default()
                        .push(op);
                }
            }

            if device_ops.is_empty() {
                eprintln!(
                    "Warning: No ready operations but {} unscheduled",
                    num_ops - scheduled_count
                );
                break;
            }

            // For each device, build the longest chain we can execute
            for (device, mut ops) in device_ops {
                if ops.is_empty() {
                    continue;
                }

                ops.sort_unstable();

                // Extend the chain: keep adding successors if they're on the same device and ready
                let mut chain = ops;
                let mut chain_set: HashSet<OperationId> = chain.iter().copied().collect();
                let mut check_idx = 0;

                // Use index-based iteration to avoid cloning
                while check_idx < chain.len() {
                    let op = chain[check_idx];

                    for &succ in &successors[op] {
                        if !scheduled[succ]
                            && !chain_set.contains(&succ)
                            && op_devices[succ] == device
                            && predecessors[succ]
                                .iter()
                                .all(|&pred| scheduled[pred] || chain_set.contains(&pred))
                        {
                            chain.push(succ);
                            chain_set.insert(succ);
                        }
                    }

                    check_idx += 1;
                }

                // Build internal dependencies with position map for O(1) lookups
                let position_map: HashMap<OperationId, usize> = chain
                    .iter()
                    .enumerate()
                    .map(|(idx, &op)| (op, idx))
                    .collect();

                let mut internal_deps = Vec::new();
                for (local_idx, &op) in chain.iter().enumerate() {
                    for &pred in &predecessors[op] {
                        if let Some(&pred_local_idx) = position_map.get(&pred) {
                            internal_deps.push((pred_local_idx, local_idx));
                        }
                    }
                }

                // Find cross-device dependencies (use HashSet to avoid duplicates)
                let mut wait_set: HashSet<(DeviceLocation, u64)> = HashSet::new();
                for &op in &chain {
                    for &pred in &predecessors[op] {
                        if scheduled[pred] {
                            let pred_device = &op_devices[pred];
                            if pred_device != &device {
                                if let Some(&sem_val) = device_semaphore_values.get(pred_device) {
                                    wait_set.insert((pred_device.clone(), sem_val));
                                }
                            }
                        }
                    }
                }
                let wait_semaphores: Vec<_> = wait_set.into_iter().collect();

                let requires_host_wait = matches!(device, DeviceLocation::CPU);
                let signal_value = current_semaphore_value;
                current_semaphore_value += 1;

                device_semaphore_values.insert(device.clone(), signal_value);

                chunks.push(ExecutionChunk {
                    device,
                    operations: chain.clone(),
                    internal_deps,
                    wait_semaphores,
                    signal_value,
                    requires_host_wait,
                });

                // Mark as scheduled and update in-degrees
                for &op in &chain {
                    scheduled[op] = true;
                    scheduled_count += 1;
                    for &succ in &successors[op] {
                        current_in_degree[succ] = current_in_degree[succ].saturating_sub(1);
                    }
                }
            }
        }

        ExecutionPlan {
            chunks,
            next_semaphore_value: current_semaphore_value,
        }
    }

    fn determine_op_device(
        &self,
        op_id: OperationId,
        device_locations: &[Option<DeviceLocation>],
    ) -> DeviceLocation {
        // Check input tensors
        for &tid in &self.operations[op_id].get_input_tensor_ids() {
            if let Some(Some(dev)) = device_locations.get(tid) {
                return dev.clone();
            }
        }

        // Check output tensors
        for &tid in &self.operations[op_id].get_output_tensor_ids() {
            if let Some(Some(dev)) = device_locations.get(tid) {
                return dev.clone();
            }
        }

        DeviceLocation::CPU
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
