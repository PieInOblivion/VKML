use onnx_extractor::DataType;

use crate::utils::onnx_autopad::calc_begin_and_end_pads;
use crate::{
    instruction,
    tensor::TensorDesc,
    utils::{OnnxAutoPad, error::VKMLError},
};

use super::{execution::LayerExecution, layer::Layer};

#[derive(Clone)]
pub struct ConvLayer {
    pub in_features: i64,  // Input channels
    pub out_features: i64, // Output channels
    pub auto_pad: OnnxAutoPad,
    pub dilations: Vec<i64>,
    pub kernel_shape: Vec<i64>,
    pub pads: Vec<i64>,
    pub strides: Vec<i64>,
    pub bias: bool,
}

impl ConvLayer {
    /*
    // TODO: Implement a conv layer constructor with the most minimal defaults possible
    pub fn new(in_features: i64, out_features: i64) -> Self {
        Self {
            in_features,
            out_features,
            auto_pad: OnnxAutoPad::Valid,
            dilations:
            kernel_shape:
            pads:
            strides:
            bias: false,
        }
    }
    */

    pub fn new_with(
        in_features: i64,
        out_features: i64,
        auto_pad: OnnxAutoPad,
        dilations: Vec<i64>,
        kernel_shape: Vec<i64>,
        pads: Vec<i64>,
        strides: Vec<i64>,
        bias: bool,
    ) -> Self {
        Self {
            in_features,
            out_features,
            auto_pad,
            dilations,
            kernel_shape,
            pads,
            strides,
            bias,
        }
    }
}

impl Layer for ConvLayer {
    fn output_shapes(
        &self,
        batch_size: i64,
        input_shapes: &[&TensorDesc],
    ) -> Result<Vec<TensorDesc>, VKMLError> {
        if input_shapes.len() != 1 {
            return Err(VKMLError::Layer(format!(
                "Conv layer requires exactly 1 input, got {}",
                input_shapes.len()
            )));
        }

        let input_shape = input_shapes[0];

        // Expect input tensor of shape [N, C, D1, D2, ..., Dn]
        let ndim = input_shape.ndim();
        if ndim < 3 {
            return Err(VKMLError::Layer(format!(
                "Conv requires input tensor with at least 3 dims (N,C,spatial...), got {:?}",
                input_shape
            )));
        }

        // spatial rank
        let spatial_rank = ndim - 2;

        // Verify input channels match
        let input_dims = input_shape.dims();
        let in_channels = input_dims[1];
        if input_dims[1] != self.in_features {
            return Err(VKMLError::Layer(format!(
                "Conv expected {} input channels, got {}",
                self.in_features, in_channels
            )));
        }

        // Gather spatial input sizes
        let mut spatial_input: Vec<i64> = Vec::with_capacity(spatial_rank);
        for i in 0..spatial_rank {
            spatial_input.push(input_dims[2 + i]);
        }

        // Compute pads using the shared helper and convert to i64
        let (pads_begin, pads_end) = calc_begin_and_end_pads(
            self.auto_pad.clone(),
            &self.pads,
            &self.kernel_shape,
            &self.strides,
            &self.dilations,
            input_shape,
        );

        // compute output spatial dims
        let mut out_spatial: Vec<i64> = Vec::with_capacity(spatial_rank);
        for i in 0..spatial_rank {
            let in_i = spatial_input[i];
            let k = self.kernel_shape.get(i).copied().unwrap_or(1);
            let s = self.strides.get(i).copied().unwrap_or(1);
            let d = self.dilations.get(i).copied().unwrap_or(1);
            let p_begin = pads_begin[i];
            let p_end = pads_end[i];

            let numerator = in_i + p_begin + p_end - d * (k - 1) - 1;
            let out_i = (numerator / s) + 1;
            out_spatial.push(out_i);
        }

        // Build output tensor dims: [batch, out_channels, spatial...]
        let mut out_dims = vec![batch_size, self.out_features];
        out_dims.extend(out_spatial.iter());

        Ok(vec![TensorDesc::new(out_dims, DataType::Float)])
    }

    fn parameter_shapes(&self, _input_shapes: &[&TensorDesc]) -> Option<(TensorDesc, TensorDesc)> {
        // weights shape: [out_channels, in_channels/group, k1, k2, ..., kn]
        let mut w_dims: Vec<i64> = Vec::with_capacity(2 + self.kernel_shape.len());
        w_dims.push(self.out_features);
        w_dims.push(self.in_features); // group not modeled here; assume 1
        for &k in &self.kernel_shape {
            w_dims.push(k);
        }

        let weights = TensorDesc::new(w_dims, DataType::Float);
        let biases = TensorDesc::new(vec![self.out_features], DataType::Float);

        Some((weights, biases))
    }

    fn parameter_count(&self, _batch_size: i64, _input_shapes: &[&TensorDesc]) -> i64 {
        // product of kernel spatial dims
        let mut kernel_prod: i64 = 1;
        if !self.kernel_shape.is_empty() {
            for &k in &self.kernel_shape {
                kernel_prod *= k;
            }
        }

        let weight_params = self.out_features * self.in_features * kernel_prod;
        let bias_params = if self.bias { self.out_features } else { 0 };

        weight_params + bias_params
    }

    fn input_requirements(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }

    fn name(&self) -> String {
        "Conv".to_string()
    }

    fn config_string(&self) -> Option<String> {
        // Compact and informative config string
        let ks = if self.kernel_shape.is_empty() {
            "[1]".to_string()
        } else {
            format!(
                "[{}]",
                self.kernel_shape
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        let ss = if self.strides.is_empty() {
            "[1]".to_string()
        } else {
            format!(
                "[{}]",
                self.strides
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        let ps = if self.pads.is_empty() {
            "[]".to_string()
        } else {
            format!(
                "[{}]",
                self.pads
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        let ds = if self.dilations.is_empty() {
            "[1]".to_string()
        } else {
            format!(
                "[{}]",
                self.dilations
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        Some(format!(
            "auto_pad={:?}, kernel={}, stride={}, dilation={}, pads={}, bias={}",
            self.auto_pad, ks, ss, ds, ps, self.bias
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
            return Err(VKMLError::Layer("Conv layer requires an input".to_string()));
        }

        let input_shape = input_shapes[0];

        // support N x C x D1 x D2 ... Dn
        if input_shape.ndim() < 3 {
            return Err(VKMLError::Layer(
                "Conv layer expects at least 3D tensor input".into(),
            ));
        }

        let input_dims = input_shape.dims();
        let in_channels = input_dims[1];
        if in_channels != self.in_features {
            return Err(VKMLError::Layer(format!(
                "Conv layer expects {} input channels, got {}",
                self.in_features, in_channels
            )));
        }

        let spatial_rank = input_shape.ndim() - 2;
        let mut spatial_input: Vec<i64> = Vec::with_capacity(spatial_rank);
        for i in 0..spatial_rank {
            spatial_input.push(input_dims[2 + i]);
        }

        let mut out_spatial: Vec<i64> = Vec::with_capacity(spatial_rank);
        // Compute pads and strides/dilations similar to output_shapes
        let mut kernel: Vec<i64> = Vec::with_capacity(spatial_rank);
        let mut stride: Vec<i64> = Vec::with_capacity(spatial_rank);
        let mut dilation: Vec<i64> = Vec::with_capacity(spatial_rank);
        for i in 0..spatial_rank {
            kernel.push(self.kernel_shape.get(i).copied().unwrap_or(1));
            stride.push(self.strides.get(i).copied().unwrap_or(1));
            dilation.push(self.dilations.get(i).copied().unwrap_or(1));
        }

        // Compute pads for the inner conv using shared helper.
        let (pads_begin, pads_end) = calc_begin_and_end_pads(
            self.auto_pad.clone(),
            &self.pads,
            &self.kernel_shape,
            &self.strides,
            &self.dilations,
            input_shape,
        );

        for i in 0..spatial_rank {
            let in_i = spatial_input[i];
            let k = kernel[i];
            let s = stride[i];
            let d = dilation[i];
            let p_begin = pads_begin[i];
            let p_end = pads_end[i];

            let numerator = in_i + p_begin + p_end - d * (k - 1) - 1;
            let out_i = (numerator / s) + 1;
            out_spatial.push(out_i);
        }

        let mut tensors = Vec::new();

        // input = 0
        tensors.push(input_shape.clone());

        // weights = 1: shape [out_channels, in_channels, k1, k2, ...]
        let mut w_dims: Vec<i64> = Vec::with_capacity(2 + self.kernel_shape.len());
        w_dims.push(self.out_features);
        w_dims.push(self.in_features);
        for &k in &self.kernel_shape {
            w_dims.push(k);
        }
        tensors.push(TensorDesc::new(w_dims, DataType::Float));

        // output = 2: shape [batch, out_channels, spatial...]
        let mut out_dims: Vec<i64> = Vec::with_capacity(2 + out_spatial.len());
        out_dims.push(batch_size);
        out_dims.push(self.out_features);
        out_dims.extend(out_spatial.iter());
        tensors.push(TensorDesc::new(out_dims, DataType::Float));

        let mut bias_idx = None;
        if self.bias {
            // bias = 3
            bias_idx = Some(tensors.len());
            tensors.push(TensorDesc::new(vec![self.out_features], DataType::Float));
        }

        // Create Conv instruction
        let instruction = instruction::conv(
            0,                         // src tensor id
            1,                         // weights tensor id
            bias_idx,                  // optional bias tensor id
            2,                         // dst tensor id
            self.auto_pad.clone(),     // auto_pad
            self.dilations.clone(),    // dilations
            1,                         // group (default 1)
            self.kernel_shape.clone(), // kernel_shape
            self.pads.clone(),         // pads
            self.strides.clone(),      // strides
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
