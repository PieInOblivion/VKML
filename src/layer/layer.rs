use crate::{dataloader::error::VKMLEngineError, tensor::tensor_desc::TensorDesc};

use super::execution::LayerExecution;

pub trait Layer {
    // Calculate the output shapes for all outputs of this layer
    fn output_shapes(
        &self,
        batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLEngineError>;

    // Get memory requirements for this layer
    fn memory_requirements(&self, input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64;

    // Whether this layer requires trainable parameters
    fn requires_parameters(&self) -> bool {
        self.parameter_count(0, &[]) > 0
    }

    // Whether this layer requires gradient tracking
    fn requires_gradients(&self) -> bool;

    // For parameterised layers, describes the required weight and bias tensors
    fn parameter_shapes(&self, input_shapes: &[&TensorDesc]) -> Option<(TensorDesc, TensorDesc)> {
        None
    }

    // Return the total number of parameters in this layer
    fn parameter_count(&self, _batch_size: usize, _input_shapes: &[&TensorDesc]) -> usize {
        0
    }

    // For graph verification, how many inputs this layer requires (min and max)
    fn input_requirements(&self) -> (usize, Option<usize>);

    // Return a string representation of the layers name
    fn name(&self) -> String;

    // Return optional configuration details for the layer
    fn config_string(&self) -> Option<String> {
        None
    }

    // Get input features
    fn in_features(&self) -> usize {
        0
    }

    // Get output features
    fn out_features(&self) -> usize {
        0
    }

    // Generate tensor descriptions, instructions, and outputs for this layer
    fn build_layer_exec(
        &self,
        batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<LayerExecution, VKMLEngineError>;
}
