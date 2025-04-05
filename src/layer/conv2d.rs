use crate::{
    dataloader::error::VKMLEngineError, instruction::factory::Instructions,
    tensor::tensor_desc::TensorDesc,
};

use super::{execution::LayerExecution, layer::Layer};

#[derive(Clone)]
pub struct Conv2DLayer {
    pub in_features: usize,  // Input channels
    pub out_features: usize, // Output channels
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub bias: bool,
}

impl Conv2DLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            padding_h: 0,
            padding_w: 0,
            bias: false,
        }
    }

    pub fn new_with(
        in_features: usize,
        out_features: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        bias: bool,
    ) -> Self {
        Self {
            in_features,
            out_features,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            bias,
        }
    }
}

impl Layer for Conv2DLayer {
    fn output_shapes(
        &self,
        batch_size: usize,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLEngineError> {
        if input_shapes.len() != 1 {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Conv2D layer requires exactly 1 input, got {}",
                input_shapes.len()
            )));
        }

        let input_shape = input_shapes[0];

        // Check if it's a 4D tensor (batch, channels, height, width)
        if input_shape.ndim() != 4 {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Conv2D requires 4D input tensor, got {:?}",
                input_shape
            )));
        }

        // Verify input channels match
        let in_channels = input_shape.to_dims()[1];
        if in_channels != self.in_features {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Conv2D expected {} input channels, got {}",
                self.in_features, in_channels
            )));
        }

        let h_in = input_shape.to_dims()[2];
        let w_in = input_shape.to_dims()[3];

        let h_out = ((h_in + 2 * self.padding_h - self.kernel_h) / self.stride_h) + 1;
        let w_out = ((w_in + 2 * self.padding_w - self.kernel_w) / self.stride_w) + 1;

        Ok(vec![TensorDesc::new(vec![
            batch_size,
            self.out_features,
            h_out,
            w_out,
        ])])
    }

    fn memory_requirements(&self, _input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64 {
        // Calculate weights size (out_channels * in_channels * kernel_h * kernel_w)
        let weights_size = (self.out_features
            * self.in_features
            * self.kernel_h
            * self.kernel_w
            * std::mem::size_of::<f32>()) as u64;

        // Calculate bias size (out_channels)
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
        let weights = TensorDesc::new(vec![
            self.out_features,
            self.in_features,
            self.kernel_h,
            self.kernel_w,
        ]);

        let biases = TensorDesc::new(vec![self.out_features]);

        Some((weights, biases))
    }

    fn parameter_count(&self, _batch_size: usize, _input_shapes: &[&TensorDesc]) -> usize {
        let weight_params = self.out_features * self.in_features * self.kernel_h * self.kernel_w;
        let bias_params = if self.bias { self.out_features } else { 0 };

        weight_params + bias_params
    }

    fn input_requirements(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }

    fn name(&self) -> String {
        "Conv2D".to_string()
    }

    fn config_string(&self) -> Option<String> {
        Some(format!(
            "kernel={}×{}, stride={}×{}, padding={}×{}, bias={}",
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.bias
        ))
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
                "Conv2D layer requires an input".to_string(),
            ));
        }

        let input_shape = input_shapes[0];

        if input_shape.ndim() != 4 {
            return Err(VKMLEngineError::VulkanLoadError(
                "Conv2D layer expects 4D tensor input".into(),
            ));
        }

        let in_channels = input_shape.to_dims()[1];
        let in_height = input_shape.to_dims()[2];
        let in_width = input_shape.to_dims()[3];

        if in_channels != self.in_features {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "Conv2D layer expects {} input channels, got {}",
                self.in_features, in_channels
            )));
        }

        let out_height = ((in_height + 2 * self.padding_h - self.kernel_h) / self.stride_h) + 1;
        let out_width = ((in_width + 2 * self.padding_w - self.kernel_w) / self.stride_w) + 1;

        let mut tensors = Vec::new();

        // input = 0
        tensors.push(input_shape.clone());

        // weights = 1
        tensors.push(TensorDesc::new(vec![
            self.out_features,
            self.in_features,
            self.kernel_h,
            self.kernel_w,
        ]));

        // output = 2
        tensors.push(TensorDesc::new(vec![
            batch_size,
            self.out_features,
            out_height,
            out_width,
        ]));

        let mut bias_idx = None;
        if self.bias {
            // bias = 3
            bias_idx = Some(tensors.len());
            tensors.push(TensorDesc::new(vec![self.out_features]));
        }

        // Create Conv2D instruction
        let instruction = Instructions::conv2d(
            0,
            1,
            bias_idx,
            2,
            (self.stride_h, self.stride_w),
            (self.padding_h, self.padding_w),
        );

        // Get input mappings using the trait method
        let input_mappings = self.map_input_tensors(input_shapes.len());

        Ok(LayerExecution {
            tensors,
            instructions: vec![instruction],
            outputs: vec![2],
            input_mappings,
        })
    }
}
