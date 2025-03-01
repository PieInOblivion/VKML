use std::collections::HashMap;

use crate::{compute::{compute_manager::ComputeTensor, location::ComputeLocation}, dataloader::error::VKMLEngineError, model::{instruction::Instruction, tensor_desc::TensorDesc}};

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
    fn output_shapes(&self, batch_size: usize, input_shapes: &[&TensorDesc]) -> Result<Vec<TensorDesc>, VKMLEngineError> {
        if input_shapes.len() != 1 {
            return Err(VKMLEngineError::VulkanLoadError(
                format!("Activation layer requires exactly 1 input, got {}", input_shapes.len())
            ));
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
            _ => None
        }
    }

    fn build_layer_exec(&self, batch_size: usize, input_shapes: &[&TensorDesc]) -> Result<LayerExecution, VKMLEngineError> {
        if input_shapes.is_empty() {
            return Err(VKMLEngineError::VulkanLoadError(
                "Activation layer requires an input".to_string()
            ));
        }
        
        let input_shape = input_shapes[0];
        
        let mut tensors = HashMap::new();
        
        tensors.insert("input".to_string(), ComputeTensor {
            desc: input_shape.clone(),
            location: ComputeLocation::Unallocated,
        });
        
        tensors.insert("output".to_string(), ComputeTensor {
            desc: input_shape.clone(),
            location: ComputeLocation::Unallocated,
        });
        
        let mut instructions = Vec::new();
        
        instructions.push(Instruction::ReadInput {
            layer_idx: 0,
            layer_tensor_idx: 0,
            dst: "input".to_string(),
        });
        
        let activation_instruction = match &self.activation_type {
            ActivationType::ReLU => Instruction::ReLU {
                src: "input".to_string(),
                dst: "output".to_string(),
            },
            ActivationType::LeakyReLU(alpha) => Instruction::LeakyReLU {
                src: "input".to_string(),
                dst: "output".to_string(),
                alpha: *alpha,
            },
            ActivationType::Sigmoid => Instruction::Sigmoid {
                src: "input".to_string(),
                dst: "output".to_string(),
            },
            ActivationType::Softmax(dim) => Instruction::Softmax {
                src: "input".to_string(),
                dst: "output".to_string(),
                dim: *dim,
            },
            ActivationType::Tanh => Instruction::Tanh {
                src: "input".to_string(),
                dst: "output".to_string(),
            },
            ActivationType::GELU => Instruction::GELU {
                src: "input".to_string(),
                dst: "output".to_string(),
            },
            ActivationType::SiLU => Instruction::SiLU {
                src: "input".to_string(),
                dst: "output".to_string(),
            },
        };
        instructions.push(activation_instruction);
        
        Ok(LayerExecution {
            tensors,
            instructions,
            outputs: vec!["output".to_string()],
        })
    }
}