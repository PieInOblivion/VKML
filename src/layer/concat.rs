use crate::{
    dataloader::error::VKMLEngineError, instruction::factory::Instructions,
    tensor::tensor_desc::TensorDesc,
};

use super::{execution::LayerExecution, layer::Layer};

pub struct ConcatLayer {
    pub dim: usize,
}

impl ConcatLayer {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Layer for ConcatLayer {
    fn name(&self) -> String {
        "Concat".to_string()
    }

    fn config_string(&self) -> Option<String> {
        Some(format!("dim={}", self.dim))
    }

    fn requires_parameters(&self) -> bool {
        false
    }

    fn requires_gradients(&self) -> bool {
        true
    }

    fn parameter_count(&self, _batch_size: usize, _input_shapes: &[&TensorDesc]) -> usize {
        0 // No parameters
    }

    fn in_features(&self) -> usize {
        0 // Dynamic based on inputs
    }

    fn out_features(&self) -> usize {
        0 // Dynamic based on inputs
    }

    fn input_requirements(&self) -> (usize, Option<usize>) {
        (2, None) // At least 2 inputs, no maximum
    }

    fn parameter_shapes(&self, _input_shapes: &[&TensorDesc]) -> Option<(TensorDesc, TensorDesc)> {
        None // No parameters
    }

    fn memory_requirements(&self, _input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64 {
        // Concat has no parameters, just activations
        let activation_bytes = output_shape.size_in_bytes() as u64;

        // Add gradient memory if needed
        let gradient_bytes = if self.requires_gradients() {
            activation_bytes
        } else {
            0
        };

        activation_bytes + gradient_bytes
    }

    fn output_shapes(
        &self,
        _batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLEngineError> {
        if input_shapes.len() < 2 {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Concat layer requires at least 2 inputs, got {}",
                input_shapes.len()
            )));
        }

        // Check that all inputs have the same number of dimensions
        let ndim = input_shapes[0].to_dims().len();
        for shape in input_shapes.iter().skip(1) {
            if shape.to_dims().len() != ndim {
                return Err(VKMLEngineError::VulkanLoadError(format!(
                    "All inputs to Concat must have same number of dimensions"
                )));
            }
        }

        // Check that concatenation dimension is valid
        if self.dim >= ndim {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Concat dimension {} out of range for {}-dimensional tensors",
                self.dim, ndim
            )));
        }

        // For all dimensions except concat_dim, sizes must match
        for d in 0..ndim {
            if d == self.dim {
                continue;
            }

            let size = input_shapes[0].to_dims()[d];
            for shape in input_shapes.iter().skip(1) {
                if shape.to_dims()[d] != size {
                    return Err(VKMLEngineError::VulkanLoadError(format!(
                        "Dimension {} must have same size for all inputs to Concat",
                        d
                    )));
                }
            }
        }

        // Calculate the output shape
        let mut output_dims = input_shapes[0].to_dims();

        // Sum the sizes along the concat dimension
        output_dims[self.dim] = input_shapes
            .iter()
            .map(|shape| shape.to_dims()[self.dim])
            .sum();

        // Create output tensor descriptor of the appropriate type
        let output_shape = TensorDesc::new(output_dims);

        Ok(vec![output_shape])
    }

    fn build_layer_exec(
        &self,
        batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<LayerExecution, VKMLEngineError> {
        if input_shapes.len() < 2 {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Concat layer requires at least 2 inputs, got {}",
                input_shapes.len()
            )));
        }

        let mut tensors = Vec::new();
        let mut input_tensor_indices = Vec::new();

        // Add input tensors
        for shape in input_shapes {
            let idx = tensors.len();
            tensors.push((*shape).clone());
            input_tensor_indices.push(idx);
        }

        // Calculate output shape
        let output_shapes = self.output_shapes(batch_size, input_shapes)?;
        let output_shape = output_shapes[0].clone();

        // Add output tensor
        let output_idx = tensors.len();
        tensors.push(output_shape);

        // Create Concat instruction
        let instruction = Instructions::concat(input_tensor_indices, output_idx, self.dim);

        // Get input mappings using the trait method
        let input_mappings = self.map_input_tensors(input_shapes.len());

        Ok(LayerExecution {
            tensors,
            instructions: vec![instruction],
            outputs: vec![output_idx],
            input_mappings,
        })
    }
}
