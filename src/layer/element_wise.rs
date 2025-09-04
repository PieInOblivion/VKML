use crate::{
    dataloader::error::VKMLError, instruction::factory::Instructions, tensor::desc::TensorDesc,
};

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
    fn output_shapes(
        &self,
        _batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLError> {
        if input_shapes.len() < 2 {
            return Err(VKMLError::VulkanLoadError(format!(
                "Element-wise operation requires at least 2 inputs, got {}",
                input_shapes.len()
            )));
        }

        // All inputs must have the same shape
        let first_shape = input_shapes[0];
        for shape in &input_shapes[1..] {
            if shape.to_dims() != first_shape.to_dims() {
                return Err(VKMLError::VulkanLoadError(format!(
                    "Element-wise operations require matching dimensions: {:?} vs {:?}",
                    first_shape.to_dims(),
                    shape.to_dims()
                )));
            }
        }

        // Output has the same shape as inputs
        Ok(vec![first_shape.clone()])
    }

    fn memory_requirements(&self, _input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64 {
        // Element-wise operations only need memory for output activations and gradients
        let activation_size = output_shape.size_in_bytes() as u64;
        activation_size * 2 // For activations and gradients
    }

    fn input_requirements(&self) -> (usize, Option<usize>) {
        (2, None) // At least 2 inputs, no upper limit
    }

    fn name(&self) -> String {
        self.operation.name()
    }

    fn build_layer_exec(
        &self,
        _batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<LayerExecution, VKMLError> {
        if input_shapes.len() < 2 {
            return Err(VKMLError::VulkanLoadError(format!(
                "Element-wise operation requires at least 2 inputs, got {}",
                input_shapes.len()
            )));
        }

        // Check that all inputs have the same shape
        let first_shape = input_shapes[0];
        for shape in &input_shapes[1..] {
            if shape.to_dims() != first_shape.to_dims() {
                return Err(VKMLError::VulkanLoadError(format!(
                    "Element-wise operations require matching dimensions: {:?} vs {:?}",
                    first_shape.to_dims(),
                    shape.to_dims()
                )));
            }
        }

        let mut tensors = Vec::new();

        // Add the first input tensor
        tensors.push(first_shape.clone());

        // Add the second input tensor
        tensors.push(first_shape.clone());

        // Add output tensor
        tensors.push(first_shape.clone());

        // Create element-wise instruction
        let element_wise_instruction = match self.operation {
            ElementWiseOperation::Add => Instructions::add(0, 1, 2),
            ElementWiseOperation::Subtract => Instructions::sub(0, 1, 2),
            ElementWiseOperation::Multiply => Instructions::mul(0, 1, 2),
            ElementWiseOperation::Divide => Instructions::div(0, 1, 2),
            ElementWiseOperation::Maximum => Instructions::max(0, 1, 2),
            ElementWiseOperation::Minimum => Instructions::min(0, 1, 2),
        };

        // Get input mappings using the trait method
        // Note: For element-wise ops, we only use the first two inputs regardless of how many are provided
        let input_mappings = self.map_input_tensors(input_shapes.len().min(2));

        Ok(LayerExecution {
            tensors,
            instructions: vec![element_wise_instruction],
            outputs: vec![2],
            input_mappings,
        })
    }
}
