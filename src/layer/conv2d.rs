use crate::{
    dataloader::error::VKMLError, instruction::factory::Instructions, tensor::desc::TensorDesc,
};

use super::{execution::LayerExecution, layer::Layer};

#[derive(Clone)]
pub struct Conv2DLayer {
    pub in_features: i64,  // Input channels
    pub out_features: i64, // Output channels
    pub kernel_h: i64,
    pub kernel_w: i64,
    pub stride_h: i64,
    pub stride_w: i64,
    pub padding_h: i64,
    pub padding_w: i64,
    pub bias: bool,
}

impl Conv2DLayer {
    pub fn new(in_features: i64, out_features: i64) -> Self {
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
        in_features: i64,
        out_features: i64,
        kernel_h: i64,
        kernel_w: i64,
        stride_h: i64,
        stride_w: i64,
        padding_h: i64,
        padding_w: i64,
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
        batch_size: i64,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLError> {
        if input_shapes.len() != 1 {
            return Err(VKMLError::VulkanLoadError(format!(
                "Conv2D layer requires exactly 1 input, got {}",
                input_shapes.len()
            )));
        }

        let input_shape = input_shapes[0];

        // Check if it's a 4D tensor (batch, channels, height, width)
        if input_shape.ndim() != 4 {
            return Err(VKMLError::VulkanLoadError(format!(
                "Conv2D requires 4D input tensor, got {:?}",
                input_shape
            )));
        }

        // Verify input channels match
        let in_channels = input_shape.to_dims()[1];
        if in_channels != self.in_features {
            return Err(VKMLError::VulkanLoadError(format!(
                "Conv2D expected {} input channels, got {}",
                self.in_features, in_channels
            )));
        }

        let h_in = input_shape.to_dims()[2];
        let w_in = input_shape.to_dims()[3];

        // Ensure kernel/stride/padding are sensible; default negatives to 1
        let k_h = if self.kernel_h <= 0 { 1 } else { self.kernel_h };
        let k_w = if self.kernel_w <= 0 { 1 } else { self.kernel_w };
        let s_h = if self.stride_h <= 0 { 1 } else { self.stride_h };
        let s_w = if self.stride_w <= 0 { 1 } else { self.stride_w };
        let p_h = if self.padding_h <= 0 {
            1
        } else {
            self.padding_h
        };
        let p_w = if self.padding_w <= 0 {
            1
        } else {
            self.padding_w
        };

        let h_out = ((h_in + 2 * p_h - k_h) / s_h) + 1;
        let w_out = ((w_in + 2 * p_w - k_w) / s_w) + 1;

        Ok(vec![TensorDesc::new(vec![
            batch_size,
            self.out_features,
            h_out,
            w_out,
        ])])
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

    fn parameter_count(&self, _batch_size: i64, _input_shapes: &[&TensorDesc]) -> i64 {
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

    fn in_features(&self) -> i64 {
        self.in_features
    }

    fn out_features(&self) -> i64 {
        self.out_features
    }

    fn build_layer_exec(
        &self,
        batch_size: i64,
        input_shapes: &[&TensorDesc],
    ) -> Result<LayerExecution, VKMLError> {
        if input_shapes.is_empty() {
            return Err(VKMLError::VulkanLoadError(
                "Conv2D layer requires an input".to_string(),
            ));
        }

        let input_shape = input_shapes[0];

        if input_shape.ndim() != 4 {
            return Err(VKMLError::VulkanLoadError(
                "Conv2D layer expects 4D tensor input".into(),
            ));
        }

        let in_channels = input_shape.to_dims()[1];
        let in_height = input_shape.to_dims()[2];
        let in_width = input_shape.to_dims()[3];

        if in_channels != self.in_features {
            return Err(VKMLError::VulkanLoadError(format!(
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
        // Prepare stride/padding for the instruction (must be usize)
        let s_h = if self.stride_h <= 0 { 1 } else { self.stride_h } as usize;
        let s_w = if self.stride_w <= 0 { 1 } else { self.stride_w } as usize;
        let p_h = if self.padding_h <= 0 {
            1
        } else {
            self.padding_h
        } as usize;
        let p_w = if self.padding_w <= 0 {
            1
        } else {
            self.padding_w
        } as usize;

        let instruction = Instructions::conv2d(0, 1, bias_idx, 2, (s_h, s_w), (p_h, p_w));

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
