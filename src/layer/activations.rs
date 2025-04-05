use crate::{
    dataloader::error::VKMLEngineError, instruction::factory::Instructions,
    tensor::tensor_desc::TensorDesc,
};

use super::{execution::LayerExecution, layer::Layer};

pub trait ActivationFunction: Clone {
    fn name(&self) -> String;
    fn to_string(&self) -> String;
}

#[derive(Clone)]
pub enum ActivationType {
    ReLU,
    LeakyReLU(f32),
    Sigmoid,
    Softmax(usize),
    Tanh,
    GELU,
    SiLU,
}

impl ActivationFunction for ActivationType {
    fn name(&self) -> String {
        match self {
            ActivationType::ReLU => "ReLU".to_string(),
            ActivationType::LeakyReLU(_) => "LeakyReLU".to_string(),
            ActivationType::Sigmoid => "Sigmoid".to_string(),
            ActivationType::Softmax(_) => "Softmax".to_string(),
            ActivationType::Tanh => "Tanh".to_string(),
            ActivationType::GELU => "GELU".to_string(),
            ActivationType::SiLU => "SiLU".to_string(),
        }
    }

    fn to_string(&self) -> String {
        match self {
            ActivationType::ReLU => "ReLU".to_string(),
            ActivationType::LeakyReLU(alpha) => format!("LeakyReLU(α={})", alpha),
            ActivationType::Sigmoid => "Sigmoid".to_string(),
            ActivationType::Softmax(dim) => format!("Softmax(dim={})", dim),
            ActivationType::Tanh => "Tanh".to_string(),
            ActivationType::GELU => "GELU".to_string(),
            ActivationType::SiLU => "SiLU".to_string(),
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
        _batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLEngineError> {
        if input_shapes.len() != 1 {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Activation layer requires exactly 1 input, got {}",
                input_shapes.len()
            )));
        }

        // Activation functions preserve input shape - return as a single-element vector
        Ok(vec![input_shapes[0].clone()])
    }

    fn memory_requirements(&self, _input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64 {
        // Activation layers only need memory for activations and gradients
        let activation_size = output_shape.size_in_bytes() as u64;
        activation_size * 2
    }

    fn requires_gradients(&self) -> bool {
        true
    }

    fn input_requirements(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }

    fn name(&self) -> String {
        self.activation_type.name()
    }

    fn config_string(&self) -> Option<String> {
        match &self.activation_type {
            ActivationType::LeakyReLU(alpha) => Some(format!("alpha={}", alpha)),
            ActivationType::Softmax(dim) => Some(format!("dim={}", dim)),
            _ => None,
        }
    }

    fn build_layer_exec(
        &self,
        _batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<LayerExecution, VKMLEngineError> {
        if input_shapes.is_empty() {
            return Err(VKMLEngineError::VulkanLoadError(
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
            ActivationType::ReLU => Instructions::relu(0, 1),
            ActivationType::LeakyReLU(alpha) => Instructions::leaky_relu(0, 1, *alpha),
            ActivationType::Sigmoid => Instructions::sigmoid(0, 1),
            ActivationType::Softmax(dim) => Instructions::softmax(0, 1, *dim),
            ActivationType::Tanh => Instructions::tanh(0, 1),
            ActivationType::GELU => Instructions::gelu(0, 1),
            ActivationType::SiLU => Instructions::silu(0, 1),
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
