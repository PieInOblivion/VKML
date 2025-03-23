use std::collections::HashMap;

use crate::{
    dataloader::error::VKMLEngineError, model::instruction::Instruction,
    tensor::tensor_desc::TensorDesc,
};

use super::{execution::LayerExecution, layer::Layer};

pub struct LinearLayer {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
}

impl LinearLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            bias: true,
        }
    }

    pub fn new_with(in_features: usize, out_features: usize, bias: bool) -> Self {
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
        batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLEngineError> {
        if input_shapes.len() != 1 {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Linear layer requires exactly 1 input, got {}",
                input_shapes.len()
            )));
        }

        let input_shape = input_shapes[0];

        // Check if it's a 2D tensor (batch, features)
        if input_shape.ndim() != 2 {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Linear layer requires matrix input, got tensor with {} dimensions",
                input_shape.ndim()
            )));
        }

        let cols = input_shape.to_dims()[1];
        if cols != self.in_features {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Linear layer expected {} input features, got {}",
                self.in_features, cols
            )));
        }

        // Output shape is [batch_size, out_features]
        Ok(vec![TensorDesc::new(vec![batch_size, self.out_features])])
    }

    fn memory_requirements(&self, _input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64 {
        // Memory for weights: in_features * out_features
        let weights_size =
            (self.in_features * self.out_features * std::mem::size_of::<f32>()) as u64;

        let bias_size = if self.bias {
            (self.out_features * std::mem::size_of::<f32>()) as u64
        } else {
            0
        };

        let activation_size = output_shape.size_in_bytes() as u64;

        let gradient_size = weights_size + bias_size + activation_size;

        weights_size + bias_size + activation_size + gradient_size
    }

    fn requires_gradients(&self) -> bool {
        true
    }

    fn parameter_shapes(&self, _input_shapes: &[&TensorDesc]) -> Option<(TensorDesc, TensorDesc)> {
        let weights = TensorDesc::new(vec![self.out_features, self.in_features]);
        let biases = TensorDesc::new(vec![self.out_features]);

        Some((weights, biases))
    }

    fn parameter_count(&self, _batch_size: usize, _input_shapes: &[&TensorDesc]) -> usize {
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

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn build_layer_exec(
        &self,
        batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<LayerExecution, VKMLEngineError> {
        if input_shapes.is_empty() {
            return Err(VKMLEngineError::VulkanLoadError(
                "LinearLayer requires at least one input".to_string(),
            ));
        }

        let input_shape = input_shapes[0]; // Use the first input shape

        if input_shape.ndim() != 2 {
            return Err(VKMLEngineError::VulkanLoadError(
                "Linear layer expects matrix input".into(),
            ));
        }

        let cols = input_shape.to_dims()[1];
        if cols != self.in_features {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Linear layer expects {} input features, got {}",
                self.in_features, cols
            )));
        }

        let mut tensors = HashMap::new();

        tensors.insert("input".to_string(), input_shape.clone());
        tensors.insert(
            "weights".to_string(),
            TensorDesc::new(vec![self.out_features, self.in_features]),
        );
        tensors.insert(
            "output".to_string(),
            TensorDesc::new(vec![batch_size, self.out_features]),
        );

        let mut instructions = Vec::new();

        instructions.push(Instruction::ReadInput {
            layer_idx: 0,
            layer_tensor_idx: 0,
            dst: "input".to_string(),
        });

        instructions.push(Instruction::MatMul {
            src1: "input".to_string(),
            src2: "weights".to_string(),
            dst: "output".to_string(),
        });

        // If using bias, add it
        if self.bias {
            tensors.insert("bias".to_string(), TensorDesc::new(vec![self.out_features]));

            instructions.push(Instruction::Add {
                src1: "output".to_string(),
                src2: "bias".to_string(),
                dst: "output".to_string(),
            });
        }

        Ok(LayerExecution {
            tensors,
            instructions,
            outputs: vec!["output".to_string()],
        })
    }
}
