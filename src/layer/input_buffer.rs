use std::collections::HashMap;

use onnx_extractor::DataType;

use crate::{tensor::TensorDesc, utils::error::VKMLError};

use super::{execution::LayerExecution, layer::Layer};

#[derive(Clone)]
pub struct InputLayer {
    pub out_features: i64,
    pub track_gradients: bool,
}

impl InputLayer {
    pub fn new(out_features: i64) -> Self {
        Self {
            out_features,
            track_gradients: false,
        }
    }

    pub fn new_with(out_features: i64, track_gradients: bool) -> Self {
        Self {
            out_features,
            track_gradients,
        }
    }
}

impl Layer for InputLayer {
    fn output_shapes(
        &self,
        batch_size: i64,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLError> {
        // Input layers ignore input_shapes since they're entry points
        if !input_shapes.is_empty() {
            return Err(VKMLError::Layer(format!(
                "InputBuffer expects 0 inputs, got {}",
                input_shapes.len()
            )));
        }

        Ok(vec![TensorDesc::new(
            vec![batch_size, self.out_features],
            DataType::Float,
        )])
    }

    fn input_requirements(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }

    fn name(&self) -> String {
        "InputBuffer".to_string()
    }

    fn config_string(&self) -> Option<String> {
        if self.track_gradients {
            Some("with_gradients=true".to_string())
        } else {
            Some("with_gradients=false".to_string())
        }
    }

    fn out_features(&self) -> i64 {
        self.out_features
    }

    fn build_layer_exec(
        &self,
        batch_size: i64,
        _input_shapes: &[&TensorDesc],
    ) -> Result<LayerExecution, VKMLError> {
        // Input layers don't need input_shapes - they create their own shapes
        let mut tensors = Vec::new();

        // output = 0
        tensors.push(TensorDesc::new(
            vec![batch_size, self.out_features],
            DataType::Float,
        ));

        // Add gradient tensor if tracking gradients
        if self.track_gradients {
            // gradients = 1
            tensors.push(TensorDesc::new(
                vec![batch_size, self.out_features],
                DataType::Float,
            ));
        }

        Ok(LayerExecution {
            tensors,
            instructions: vec![],
            outputs: vec![0],
            input_mappings: HashMap::new(),
        })
    }
}
