use std::collections::HashMap;

use crate::{dataloader::error::VKMLEngineError, model::instruction::Instruction, tensor::tensor_desc::TensorDesc};

use super::{execution::LayerExecution, layer::Layer};

#[derive(Clone)]
pub enum ElementWiseOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
}

impl ElementWiseOperation {
    fn name(&self) -> String {
        match self {
            ElementWiseOperation::Add => "Add".to_string(),
            ElementWiseOperation::Subtract => "Subtract".to_string(),
            ElementWiseOperation::Multiply => "Multiply".to_string(),
            ElementWiseOperation::Divide => "Divide".to_string(),
            ElementWiseOperation::Maximum => "Maximum".to_string(),
            ElementWiseOperation::Minimum => "Minimum".to_string(),
        }
    }
}

#[derive(Clone)]
pub struct ElementWiseLayer {
    pub operation: ElementWiseOperation,
}

impl ElementWiseLayer {
    pub fn new(operation: ElementWiseOperation) -> Self {
        Self { operation }
    }
}

impl Layer for ElementWiseLayer {
    fn output_shapes(&self, _batch_size: usize, input_shapes: &[&TensorDesc]) -> Result<Vec<TensorDesc>, VKMLEngineError> {
        if input_shapes.len() < 2 {
            return Err(VKMLEngineError::VulkanLoadError(
                format!("Element-wise operation requires at least 2 inputs, got {}", input_shapes.len())
            ));
        }
        
        // All inputs must have the same shape
        let first_shape = input_shapes[0];
        for shape in &input_shapes[1..] {
            if shape.to_dims() != first_shape.to_dims() {
                return Err(VKMLEngineError::VulkanLoadError(
                    format!("Element-wise operations require matching dimensions: {:?} vs {:?}", 
                           first_shape.to_dims(), shape.to_dims())
                ));
            }
        }
        
        // Output has the same shape as inputs
        Ok(vec![first_shape.clone()])
    }
    
    fn memory_requirements(&self, _input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64 {
        // Element-wise operations only need memory for output activations and gradients
        let activation_size = output_shape.size_in_bytes() as u64;
        activation_size * 2  // For activations and gradients
    }
    
    fn requires_gradients(&self) -> bool {
        true
    }
    
    fn input_requirements(&self) -> (usize, Option<usize>) {
        (2, None)  // At least 2 inputs, no upper limit
    }

    fn name(&self) -> String {
        self.operation.name()
    }

    fn build_layer_exec(&self, batch_size: usize, input_shapes: &[&TensorDesc]) -> Result<LayerExecution, VKMLEngineError> {
        if input_shapes.len() < 2 {
            return Err(VKMLEngineError::VulkanLoadError(
                format!("Element-wise operation requires at least 2 inputs, got {}", input_shapes.len())
            ));
        }
        
        // Check that all inputs have the same shape
        let first_shape = input_shapes[0];
        for shape in &input_shapes[1..] {
            if shape.to_dims() != first_shape.to_dims() {
                return Err(VKMLEngineError::VulkanLoadError(
                    format!("Element-wise operations require matching dimensions: {:?} vs {:?}", 
                           first_shape.to_dims(), shape.to_dims())
                ));
            }
        }
        
        let mut tensors = HashMap::new();
        
        tensors.insert("input0".to_string(), first_shape.clone());
        tensors.insert("input1".to_string(), first_shape.clone());
        tensors.insert("output".to_string(), first_shape.clone());
        
        let mut instructions = Vec::new();
        
        instructions.push(Instruction::ReadInput {
            layer_idx: 0,
            layer_tensor_idx: 0,
            dst: "input0".to_string(),
        });
        
        instructions.push(Instruction::ReadInput {
            layer_idx: 1,
            layer_tensor_idx: 0,
            dst: "input1".to_string(),
        });
        
        let element_wise_instruction = match self.operation {
            ElementWiseOperation::Add => Instruction::Add {
                src1: "input0".to_string(),
                src2: "input1".to_string(),
                dst: "output".to_string(),
            },
            ElementWiseOperation::Subtract => Instruction::Sub {
                src1: "input0".to_string(),
                src2: "input1".to_string(),
                dst: "output".to_string(),
            },
            ElementWiseOperation::Multiply => Instruction::Mul {
                src1: "input0".to_string(),
                src2: "input1".to_string(),
                dst: "output".to_string(),
            },
            ElementWiseOperation::Divide => Instruction::Div {
                src1: "input0".to_string(),
                src2: "input1".to_string(),
                dst: "output".to_string(),
            },
            ElementWiseOperation::Maximum => Instruction::Max {
                src1: "input0".to_string(),
                src2: "input1".to_string(),
                dst: "output".to_string(),
            },
            ElementWiseOperation::Minimum => Instruction::Min {
                src1: "input0".to_string(),
                src2: "input1".to_string(),
                dst: "output".to_string(),
            },
        };
        
        instructions.push(element_wise_instruction);
        
        Ok(LayerExecution {
            tensors,
            instructions,
            outputs: vec!["output".to_string()],
        })
    }
}