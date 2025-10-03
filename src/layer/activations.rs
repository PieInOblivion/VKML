use crate::{instruction, tensor::desc::TensorDesc, utils::error::VKMLError};

use super::{execution::LayerExecution, layer::Layer};

pub trait ActivationFunction: Clone {
    fn name(&self) -> String;
}

#[derive(Clone)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Softmax(Option<i64>),
}

impl ActivationFunction for ActivationType {
    fn name(&self) -> String {
        match self {
            ActivationType::ReLU => "ReLU".to_string(),
            ActivationType::Sigmoid => "Sigmoid".to_string(),
            ActivationType::Softmax(axis) => match axis {
                Some(a) => format!("Softmax(axis={})", a),
                None => "Softmax(axis=-1)".to_string(),
            },
        }
    }
}

// ReLU, LeakyReLU, Sigmoid, Softmax, Tanh, GELU, SiLU
#[derive(Clone)]
pub struct ActivationLayer {
    pub activation_type: ActivationType,
}

impl ActivationLayer {
    pub fn new(activation_type: ActivationType) -> Self {
        Self { activation_type }
    }
}

impl Layer for ActivationLayer {
    fn output_shapes(
        &self,
        _batch_size: i64,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLError> {
        if input_shapes.len() != 1 {
            return Err(VKMLError::VulkanError(format!(
                "Activation layer requires exactly 1 input, got {}",
                input_shapes.len()
            )));
        }

        // Activation functions preserve input shape - return as a single-element vector
        Ok(vec![input_shapes[0].clone()])
    }

    fn input_requirements(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }

    fn name(&self) -> String {
        self.activation_type.name()
    }

    fn config_string(&self) -> Option<String> {
        match &self.activation_type {
            ActivationType::Softmax(axis) => Some(match axis {
                Some(a) => format!("axis={}", a),
                None => "axis=-1".to_string(),
            }),
            _ => None,
        }
    }

    fn build_layer_exec(
        &self,
        _batch_size: i64,
        input_shapes: &[&TensorDesc],
    ) -> Result<LayerExecution, VKMLError> {
        if input_shapes.is_empty() {
            return Err(VKMLError::VulkanError(
                "Activation layer requires an input".to_string(),
            ));
        }

        let input_shape = input_shapes[0];
        let mut tensors = Vec::new();

        // input = 0
        tensors.push(input_shape.clone());

        // output = 1
        tensors.push(input_shape.clone());

        let activation_instruction = match &self.activation_type {
            ActivationType::ReLU => instruction::relu(0, 1),
            ActivationType::Sigmoid => instruction::sigmoid(0, 1),
            ActivationType::Softmax(axis) => instruction::softmax(0, 1, *axis),
        };

        // Get input mappings using the trait method
        let input_mappings = self.map_input_tensors(input_shapes.len());

        Ok(LayerExecution {
            tensors,
            instructions: vec![activation_instruction],
            outputs: vec![1],
            input_mappings,
        })
    }
}
