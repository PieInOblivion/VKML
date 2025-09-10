use crate::{dataloader::error::VKMLError, instruction, tensor::desc::TensorDesc};

use super::{execution::LayerExecution, layer::Layer};

pub struct LinearLayer {
    pub in_features: i64,
    pub out_features: i64,
    pub bias: bool,
}

impl LinearLayer {
    pub fn new(in_features: i64, out_features: i64) -> Self {
        Self {
            in_features,
            out_features,
            bias: true,
        }
    }

    pub fn new_with(in_features: i64, out_features: i64, bias: bool) -> Self {
        Self {
            in_features,
            out_features,
            bias,
        }
    }
}

impl Layer for LinearLayer {
    fn output_shapes(
        &self,
        batch_size: i64,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLError> {
        if input_shapes.len() != 1 {
            return Err(VKMLError::VulkanLoadError(format!(
                "Linear layer requires exactly 1 input, got {}",
                input_shapes.len()
            )));
        }

        let input_shape = input_shapes[0];

        // Check if it's a 2D tensor (batch, features)
        if input_shape.ndim() != 2 {
            return Err(VKMLError::VulkanLoadError(format!(
                "Linear layer requires matrix input, got tensor with {} dimensions",
                input_shape.ndim()
            )));
        }

        let cols = input_shape.to_dims()[1];
        if cols != self.in_features {
            return Err(VKMLError::VulkanLoadError(format!(
                "Linear layer expected {} input features, got {}",
                self.in_features, cols
            )));
        }

        // Output shape is [batch_size, out_features]
        Ok(vec![TensorDesc::new(vec![batch_size, self.out_features])])
    }

    fn parameter_shapes(&self, _input_shapes: &[&TensorDesc]) -> Option<(TensorDesc, TensorDesc)> {
        let weights = TensorDesc::new(vec![self.out_features, self.in_features]);
        let biases = TensorDesc::new(vec![self.out_features]);

        Some((weights, biases))
    }

    fn parameter_count(&self, _batch_size: i64, _input_shapes: &[&TensorDesc]) -> i64 {
        let weight_params = self.in_features * self.out_features;
        let bias_params = if self.bias { self.out_features } else { 0 };

        weight_params + bias_params
    }

    fn input_requirements(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }

    fn name(&self) -> String {
        "Linear".to_string()
    }

    fn config_string(&self) -> Option<String> {
        if self.bias {
            Some("bias=true".to_string())
        } else {
            Some("bias=false".to_string())
        }
    }

    fn in_features(&self) -> i64 {
        self.in_features
    }

    fn out_features(&self) -> i64 {
        self.out_features
    }

    fn build_layer_exec(
        &self,
        batch_size: i64,
        input_shapes: &[&TensorDesc],
    ) -> Result<LayerExecution, VKMLError> {
        if input_shapes.is_empty() {
            return Err(VKMLError::VulkanLoadError(
                "LinearLayer requires at least one input".to_string(),
            ));
        }

        let input_shape = input_shapes[0]; // Use the first input shape

        if input_shape.ndim() != 2 {
            return Err(VKMLError::VulkanLoadError(
                "Linear layer expects matrix input".into(),
            ));
        }

        let cols = input_shape.to_dims()[1];
        if cols != self.in_features {
            return Err(VKMLError::VulkanLoadError(format!(
                "Linear layer expects {} input features, got {}",
                self.in_features, cols
            )));
        }

        let mut tensors = Vec::new();
        let mut instructions = Vec::new();

        // input = 0
        tensors.push(input_shape.clone());

        // weights = 1
        tensors.push(TensorDesc::new(vec![self.out_features, self.in_features]));

        // output = 2
        tensors.push(TensorDesc::new(vec![batch_size, self.out_features]));

        // Create MatMul instruction
        instructions.push(instruction::matmul(0, 1, 2));

        // If using bias, add it
        if self.bias {
            // bias = 3
            tensors.push(TensorDesc::new(vec![self.out_features]));

            // TODO: Add back add_inplace instruction
            //instructions.push(instruction::add_inplace(2, 3));
        }

        // Get input mappings using the trait method
        let input_mappings = self.map_input_tensors(input_shapes.len());

        Ok(LayerExecution {
            tensors,
            instructions,
            outputs: vec![2],
            input_mappings,
        })
    }
}
