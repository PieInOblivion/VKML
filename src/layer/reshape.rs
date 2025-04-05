use crate::{
    dataloader::error::VKMLEngineError, instruction::factory::Instructions,
    tensor::tensor_desc::TensorDesc,
};

use super::{execution::LayerExecution, layer::Layer};

#[derive(Clone)]
pub struct ReshapeLayer {
    target_shape: TensorDesc, // Store directly as TensorDesc
}

impl ReshapeLayer {
    pub fn new(target_shape: TensorDesc) -> Self {
        Self { target_shape }
    }

    pub fn flatten() -> Self {
        // Create a special shape [0, 0] that indicates flatten
        Self {
            target_shape: TensorDesc::new(vec![0, 0]),
        }
    }

    // Helper to check if this is a flatten operation
    fn is_flatten(&self) -> bool {
        let dims = self.target_shape.to_dims();
        dims.len() == 2 && dims[0] == 0 && dims[1] == 0
    }
}

impl Layer for ReshapeLayer {
    fn output_shapes(
        &self,
        batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLEngineError> {
        if input_shapes.len() != 1 {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Reshape layer requires exactly 1 input, got {}",
                input_shapes.len()
            )));
        }

        let input_shape = input_shapes[0];
        let input_elements = input_shape.num_elements();

        // Handle flatten specially
        if self.is_flatten() {
            if input_elements % batch_size != 0 {
                return Err(VKMLEngineError::VulkanLoadError(format!(
                    "Cannot flatten {} elements into batches of size {}, not evenly divisible",
                    input_elements, batch_size
                )));
            }

            return Ok(vec![TensorDesc::new(vec![
                batch_size,
                input_elements / batch_size,
            ])]);
        }

        // Get target dimensions
        let target_dims = self.target_shape.to_dims();

        // Count zeros (dimensions to be inferred)
        let zeros = target_dims.iter().filter(|&&d| d == 0).count();

        if zeros == 0 {
            // No inference needed, just check total elements
            let total_new = target_dims.iter().product::<usize>();
            if total_new != input_elements {
                return Err(VKMLEngineError::VulkanLoadError(format!(
                    "Cannot reshape {} elements into shape with {} elements",
                    input_elements, total_new
                )));
            }

            Ok(vec![self.target_shape.clone()])
        } else {
            // Use dimension inference
            let mut new_dims = target_dims.clone();

            // One dimension to infer
            if zeros == 1 {
                let known_product: usize = new_dims.iter().filter(|&&d| d != 0).product();

                if input_elements % known_product != 0 {
                    return Err(VKMLEngineError::VulkanLoadError(format!(
                        "Cannot reshape {} elements: not divisible by product of known dimensions ({})",
                        input_elements, known_product
                    )));
                }

                let inferred = input_elements / known_product;

                // Replace the zero with the inferred value
                for dim in &mut new_dims {
                    if *dim == 0 {
                        *dim = inferred;
                        break;
                    }
                }

                Ok(vec![TensorDesc::new(new_dims)])
            } else {
                return Err(VKMLEngineError::VulkanLoadError(
                    "At most one dimension can be inferred (set to 0) in reshape".to_string(),
                ));
            }
        }
    }

    fn memory_requirements(&self, _input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64 {
        output_shape.size_in_bytes() as u64
    }

    fn requires_gradients(&self) -> bool {
        true
    }

    fn input_requirements(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }

    fn name(&self) -> String {
        "Reshape".to_string()
    }

    fn config_string(&self) -> Option<String> {
        if self.is_flatten() {
            Some("flatten".to_string())
        } else {
            let shape_str = self
                .target_shape
                .to_dims()
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("×");

            Some(format!("target_shape={}", shape_str))
        }
    }

    fn out_features(&self) -> usize {
        if self.is_flatten() {
            0 // Unknown until we have input shape
        } else {
            self.target_shape.num_elements()
        }
    }

    fn build_layer_exec(
        &self,
        batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<LayerExecution, VKMLEngineError> {
        if input_shapes.is_empty() {
            return Err(VKMLEngineError::VulkanLoadError(
                "Reshape layer requires an input".to_string(),
            ));
        }

        let input_shape = input_shapes[0];
        let mut tensors = Vec::new();

        // input = 0
        tensors.push(input_shape.clone());

        let output_shapes = self.output_shapes(batch_size, &[input_shape])?;
        let output_shape = output_shapes[0].clone();

        // output = 1
        tensors.push(output_shape.clone());

        // Create Reshape instruction
        let instruction = Instructions::reshape(0, 1, output_shape);

        // Get input mappings using the trait method
        let input_mappings = self.map_input_tensors(input_shapes.len());

        Ok(LayerExecution {
            tensors,
            instructions: vec![instruction],
            outputs: vec![1],
            input_mappings,
        })
    }
}
