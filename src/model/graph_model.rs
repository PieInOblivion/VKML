use std::collections::{HashMap, HashSet};

use crate::{dataloader::error::VKMLError, layer::layer::Layer, tensor::desc::TensorDesc};

use super::{
    layer_connection::{LayerConnection, LayerId},
    weight_init::WeightInit,
};

pub struct GraphModel {
    pub batch_size: usize,
    pub weight_init: WeightInit,
    pub layers: HashMap<LayerId, GraphModelLayer>,
    pub verified: Option<GraphVerifiedData>,
}

pub struct GraphModelLayer {
    pub id: LayerId,
    pub layer: Box<dyn Layer>,
    pub weight_init: Option<WeightInit>,

    pub input_connections: Vec<LayerConnection>,
    pub output_connections: Vec<LayerConnection>,
}

pub struct GraphVerifiedData {
    pub entry_points: Vec<LayerId>,
    pub exit_points: Vec<LayerId>,
    pub execution_order: Vec<LayerId>,
}

impl GraphModel {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            weight_init: WeightInit::He,
            layers: HashMap::new(),
            verified: None,
        }
    }

    pub fn new_with(batch_size: usize, weight_init: WeightInit) -> Self {
        Self {
            batch_size,
            weight_init,
            layers: HashMap::new(),
            verified: None,
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) -> LayerId {
        let id = self.next_available_id();

        // Only connect if this isn't an input layer
        let input_connections = if !self.layers.is_empty() && layer.input_requirements().0 > 0 {
            // Find the most recently added layer ID
            let prev_id = (0..id)
                .rev()
                .find(|&prev_id| self.layers.contains_key(&prev_id));

            match prev_id {
                Some(prev_id) => vec![LayerConnection::DefaultOutput(prev_id)],
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };

        self.add_layer_with(id, layer, input_connections, None)
    }

    pub fn add_layers(&mut self, layers: Vec<Box<dyn Layer>>) -> Vec<LayerId> {
        let mut ids = Vec::new();
        for layer in layers {
            let id = self.add_layer(layer);
            ids.push(id);
        }
        ids
    }

    pub fn add_layer_with(
        &mut self,
        id: LayerId,
        layer: Box<dyn Layer>,
        input_connections: Vec<LayerConnection>,
        weight_init: Option<WeightInit>,
    ) -> LayerId {
        // Update connections in the related layers
        for connection in &input_connections {
            let input_id = connection.get_layerid();

            if let Some(input_layer) = self.layers.get_mut(&input_id) {
                // Check if this layer is already an output for the input layer
                let already_connected = input_layer
                    .output_connections
                    .iter()
                    .any(|conn| conn.get_layerid() == id);

                if !already_connected {
                    // Add this layer as an output connection with default output
                    input_layer
                        .output_connections
                        .push(LayerConnection::DefaultOutput(id));
                }
            }
        }

        let layer = GraphModelLayer {
            id,
            layer,
            weight_init,
            input_connections,
            output_connections: Vec::new(),
        };

        self.layers.insert(id, layer);
        id
    }

    pub fn next_available_id(&self) -> LayerId {
        let mut id = 0;
        while self.layers.contains_key(&id) {
            id += 1;
        }
        id
    }

    pub fn verify(&mut self) -> Result<(), VKMLError> {
        // Identify input layers as those with no incoming connections
        let input_layer_ids: Vec<LayerId> = self
            .layers
            .values()
            .filter(|layer| layer.input_connections.is_empty())
            .map(|layer| layer.id)
            .collect();

        // There should be at least one input layer
        if input_layer_ids.is_empty() {
            return Err(VKMLError::VulkanLoadError(
                "Model must have at least one input layer".into(),
            ));
        }

        // All input layers should have no inputs
        let invalid_input_layers: Vec<LayerId> = self
            .layers
            .values()
            .filter(|layer| {
                layer.layer.input_requirements().0 == 0 && !layer.input_connections.is_empty()
            })
            .map(|layer| layer.id)
            .collect();

        if !invalid_input_layers.is_empty() {
            return Err(VKMLError::VulkanLoadError(format!(
                "Input layers cannot have inputs themselves: {:?}",
                invalid_input_layers
            )));
        }

        // Find exit points (layers with no outputs)
        let exit_points: Vec<LayerId> = self
            .layers
            .values()
            .filter(|layer| layer.output_connections.is_empty())
            .map(|layer| layer.id)
            .collect();

        if exit_points.is_empty() {
            return Err(VKMLError::VulkanLoadError(
                "Model has no exit points".into(),
            ));
        }

        // Verify that all referenced layers exist and output indices are valid
        for layer in self.layers.values() {
            // Check input connections
            for connection in &layer.input_connections {
                let input_id = connection.get_layerid();

                // Check that the referenced layer exists
                if !self.layers.contains_key(&input_id) {
                    return Err(VKMLError::VulkanLoadError(format!(
                        "Layer {} references non-existent input layer {}",
                        layer.id, input_id
                    )));
                }

                // Get the referenced layer
                let input_layer = self.layers.get(&input_id).unwrap();

                // Verify output index by checking the number of outputs
                let output_idx = connection.get_outputidx();

                // Get input requirements for the source layer
                let (min_inputs, _) = input_layer.layer.input_requirements();

                // Create appropriate empty vector for input layers
                let empty_vec: Vec<&TensorDesc> = Vec::new();

                // Check output count using output_shapes without creating dummy inputs
                match input_layer.layer.output_shapes(1, &empty_vec) {
                    Ok(shapes) => {
                        if output_idx >= shapes.len() {
                            return Err(VKMLError::VulkanLoadError(format!(
                                "Layer {} requests output {} from layer {}, but it only has {} outputs",
                                layer.id,
                                output_idx,
                                input_id,
                                shapes.len()
                            )));
                        }
                    }
                    Err(e) => {
                        // If this is an input layer, it's expected to work with empty inputs
                        if min_inputs == 0 {
                            return Err(VKMLError::VulkanLoadError(format!(
                                "Input layer {} failed to validate outputs: {}",
                                input_id, e
                            )));
                        }

                        // For non-input layers, this is expected since we're providing empty inputs
                        // Instead of causing an error, we'll assume they have at least one output
                        if output_idx > 0 {
                            return Err(VKMLError::VulkanLoadError(format!(
                                "Layer {} requests output {} from layer {}, but we can only validate index 0",
                                layer.id, output_idx, input_id
                            )));
                        }
                    }
                }
            }

            // Check output connections
            for connection in &layer.output_connections {
                let output_id = connection.get_layerid();

                if !self.layers.contains_key(&output_id) {
                    return Err(VKMLError::VulkanLoadError(format!(
                        "Layer {} references non-existent output layer {}",
                        layer.id, output_id
                    )));
                }
            }
        }

        // Verify bidirectional consistency of connections
        for layer in self.layers.values() {
            for out_connection in &layer.output_connections {
                let output_id = out_connection.get_layerid();

                if let Some(output_layer) = self.layers.get(&output_id) {
                    // Check if the output layer lists this layer as an input
                    let is_connected = output_layer
                        .input_connections
                        .iter()
                        .any(|conn| conn.get_layerid() == layer.id);

                    if !is_connected {
                        return Err(VKMLError::VulkanLoadError(format!(
                            "Connection inconsistency: Layer {} lists {} as output, but {} does not list {} as input",
                            layer.id, output_id, output_id, layer.id
                        )));
                    }
                }
            }

            for in_connection in &layer.input_connections {
                let input_id = in_connection.get_layerid();

                if let Some(input_layer) = self.layers.get(&input_id) {
                    // Check if the input layer lists this layer as an output
                    let is_connected = input_layer
                        .output_connections
                        .iter()
                        .any(|conn| conn.get_layerid() == layer.id);

                    if !is_connected {
                        return Err(VKMLError::VulkanLoadError(format!(
                            "Connection inconsistency: Layer {} lists {} as input, but {} does not list {} as output",
                            layer.id, input_id, input_id, layer.id
                        )));
                    }
                }
            }
        }

        // Verify that non-input layers have at least one input connection
        let non_input_layers_without_inputs: Vec<LayerId> = self
            .layers
            .values()
            .filter(|layer| {
                layer.layer.input_requirements().0 > 0 && layer.input_connections.is_empty()
            })
            .map(|layer| layer.id)
            .collect();

        if !non_input_layers_without_inputs.is_empty() {
            return Err(VKMLError::VulkanLoadError(format!(
                "Non-input layers without inputs: {:?}",
                non_input_layers_without_inputs
            )));
        }

        // Verify input layer counts match requirements
        for layer in self.layers.values() {
            let (min_inputs, max_inputs) = layer.layer.input_requirements();
            let actual_inputs = layer.input_connections.len();

            if actual_inputs < min_inputs {
                return Err(VKMLError::VulkanLoadError(format!(
                    "Layer {} requires at least {} inputs, but has {}",
                    layer.id, min_inputs, actual_inputs
                )));
            }

            if let Some(max) = max_inputs {
                if actual_inputs > max {
                    return Err(VKMLError::VulkanLoadError(format!(
                        "Layer {} requires at most {} inputs, but has {}",
                        layer.id, max, actual_inputs
                    )));
                }
            }
        }

        // Detect cycles using a depth-first search
        if self.has_cycle() {
            return Err(VKMLError::VulkanLoadError("Model contains cycles".into()));
        }

        // Generate execution order (topological sort)
        let execution_order = self.topological_sort()?;

        // Verify that execution order includes all layers
        if execution_order.len() != self.layers.len() {
            return Err(VKMLError::VulkanLoadError(format!(
                "Execution order has {} layers but model has {} layers",
                execution_order.len(),
                self.layers.len()
            )));
        }

        self.verified = Some(GraphVerifiedData {
            entry_points: input_layer_ids,
            exit_points,
            execution_order,
        });

        Ok(())
    }

    fn topological_sort(&self) -> Result<Vec<LayerId>, VKMLError> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut temp = HashSet::new();

        // Visit each node
        for &id in self.layers.keys() {
            if !visited.contains(&id) && !temp.contains(&id) {
                self.visit_node(id, &mut visited, &mut temp, &mut result)?;
            }
        }

        // Reverse the result to get the correct execution order
        result.reverse();
        Ok(result)
    }

    fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for &id in self.layers.keys() {
            if !visited.contains(&id) {
                if self.is_cyclic_util(id, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        false
    }

    fn is_cyclic_util(
        &self,
        id: LayerId,
        visited: &mut HashSet<LayerId>,
        rec_stack: &mut HashSet<LayerId>,
    ) -> bool {
        visited.insert(id);
        rec_stack.insert(id);

        if let Some(layer) = self.layers.get(&id) {
            for connection in &layer.output_connections {
                let next_id = connection.get_layerid();

                if !visited.contains(&next_id) {
                    if self.is_cyclic_util(next_id, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(&next_id) {
                    return true;
                }
            }
        }

        rec_stack.remove(&id);
        false
    }

    fn visit_node(
        &self,
        id: LayerId,
        visited: &mut HashSet<LayerId>,
        temp: &mut HashSet<LayerId>,
        result: &mut Vec<LayerId>,
    ) -> Result<(), VKMLError> {
        if temp.contains(&id) {
            return Err(VKMLError::VulkanLoadError(format!(
                "Cycle detected involving layer {}",
                id
            )));
        }

        if visited.contains(&id) {
            return Ok(());
        }

        temp.insert(id);

        if let Some(layer) = self.layers.get(&id) {
            for connection in &layer.output_connections {
                let next_id = connection.get_layerid();
                self.visit_node(next_id, visited, temp, result)?;
            }
        }

        temp.remove(&id);
        visited.insert(id);
        result.push(id);

        Ok(())
    }
}
