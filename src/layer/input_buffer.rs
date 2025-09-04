use std::collections::HashMap;

use crate::{dataloader::error::VKMLError, tensor::desc::TensorDesc};

use super::{execution::LayerExecution, layer::Layer};

#[derive(Clone)]
pub struct InputLayer {
    pub out_features: usize,
    pub track_gradients: bool,
}

impl InputLayer {
    pub fn new(out_features: usize) -> Self {
        Self {
            out_features,
            track_gradients: false,
        }
    }

    pub fn new_with(out_features: usize, track_gradients: bool) -> Self {
        Self {
            out_features,
            track_gradients,
        }
    }
}

impl Layer for InputLayer {
    fn output_shapes(
        &self,
        batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLError> {
        // Input layers ignore input_shapes since they're entry points
        if !input_shapes.is_empty() {
            return Err(VKMLError::VulkanLoadError(format!(
                "InputBuffer expects 0 inputs, got {}",
                input_shapes.len()
            )));
        }

        Ok(vec![TensorDesc::new(vec![batch_size, self.out_features])])
    }

    fn memory_requirements(&self, _input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64 {
        // Only need memory for activations
        let activation_memory = output_shape.size_in_bytes() as u64;

        let gradient_memory = if self.track_gradients {
            output_shape.size_in_bytes() as u64
        } else {
            0
        };

        activation_memory + gradient_memory
    }

    fn input_requirements(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }

    fn name(&self) -> String {
        "InputBuffer".to_string()
    }

    fn config_string(&self) -> Option<String> {
        if self.track_gradients {
            Some(format!("with_gradients=true"))
        } else {
            Some(format!("with_gradients=false"))
        }
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn build_layer_exec(
        &self,
        batch_size: usize,
        _input_shapes: &[&TensorDesc],
    ) -> Result<LayerExecution, VKMLError> {
        // Input layers don't need input_shapes - they create their own shapes
        let mut tensors = Vec::new();

        // output = 0
        tensors.push(TensorDesc::new(vec![batch_size, self.out_features]));

        // Add gradient tensor if tracking gradients
        if self.track_gradients {
            // gradients = 1
            tensors.push(TensorDesc::new(vec![batch_size, self.out_features]));
        }

        Ok(LayerExecution {
            tensors,
            instructions: vec![],
            outputs: vec![0],
            input_mappings: HashMap::new(),
        })
    }
}
