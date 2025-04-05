use crate::{
    dataloader::error::VKMLEngineError,
    gpu::vk_gpu::GPU,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use ash::vk;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use super::instruction::Instruction;

#[derive(Clone)]
pub struct Conv2DInstruction {
    pub src: TensorId,
    pub weights: TensorId,
    pub bias: Option<TensorId>,
    pub dst: TensorId,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl Debug for Conv2DInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Conv2D(src={}, weights={}, bias={:?}, dst={}, stride={:?}, padding={:?})",
            self.src, self.weights, self.bias, self.dst, self.stride, self.padding
        )
    }
}

impl Instruction for Conv2DInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        let mut inputs = vec![self.src, self.weights];
        if let Some(bias) = self.bias {
            inputs.push(bias);
        }
        inputs
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if !new_inputs.is_empty() {
            self.src = new_inputs[0];
        }

        if new_inputs.len() > 1 {
            self.weights = new_inputs[1];
        }

        if new_inputs.len() > 2 && self.bias.is_some() {
            self.bias = Some(new_inputs[2]);
        }

        if !new_outputs.is_empty() {
            self.dst = new_outputs[0];
        }
    }

    fn create_command_buffer(
        &self,
        gpu: &GPU,
        command_buffer: vk::CommandBuffer,
        tensor_graph: &TensorGraph,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let src_mem = tensor_graph.get_gpu_memory_or_panic(&self.src);
        let weights_mem = tensor_graph.get_gpu_memory_or_panic(&self.weights);
        let dst_mem = tensor_graph.get_gpu_memory_or_panic(&self.dst);

        let src_tensor = tensor_graph.tensors.get(*&self.src).unwrap();
        let weights_tensor = tensor_graph.tensors.get(*&self.weights).unwrap();
        let dst_tensor = tensor_graph.tensors.get(*&self.dst).unwrap();

        let bias_mem = self
            .bias
            .as_ref()
            .map(|bias_id| tensor_graph.get_gpu_memory_or_panic(bias_id));

        gpu.create_conv2d_command_buffer(
            command_buffer,
            src_mem,
            weights_mem,
            bias_mem,
            dst_mem,
            src_tensor,
            weights_tensor,
            dst_tensor,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
        )
    }

    fn execute_cpu(&self, tensor_graph: &mut TensorGraph) -> Result<(), VKMLEngineError> {
        let src_data = tensor_graph.tensors[self.src].data.borrow_cpu_data()?;
        let weights_data = tensor_graph.tensors[self.weights].data.borrow_cpu_data()?;
        let mut dst_data = tensor_graph.tensors[self.dst].data.borrow_mut_cpu_data()?;

        // Check optional bias tensor
        let bias_data = if let Some(bias_id) = self.bias {
            Some(tensor_graph.tensors[bias_id].data.borrow_cpu_data()?)
        } else {
            None
        };

        let src_tensor = &tensor_graph.tensors[self.src];
        let weights_tensor = &tensor_graph.tensors[self.weights];
        let dst_tensor = &tensor_graph.tensors[self.dst];

        let src_dims = src_tensor.desc.to_dims();
        let weight_dims = weights_tensor.desc.to_dims();
        let dst_dims = dst_tensor.desc.to_dims();

        if src_dims.len() != 4 || weight_dims.len() != 4 || dst_dims.len() != 4 {
            return Err(VKMLEngineError::VulkanLoadError(
                "Conv2D requires 4D tensors for input, weights, and output".to_string(),
            ));
        }

        let batch_size = src_dims[0];
        let in_channels = src_dims[1];
        let in_height = src_dims[2];
        let in_width = src_dims[3];

        let out_channels = weight_dims[0];
        let filter_in_channels = weight_dims[1];
        let kernel_h = weight_dims[2];
        let kernel_w = weight_dims[3];

        let out_height = dst_dims[2];
        let out_width = dst_dims[3];

        // Verify output dimensions
        if batch_size != dst_dims[0] || out_channels != dst_dims[1] {
            return Err(VKMLEngineError::ShapeMismatch(format!(
                "Output dimensions mismatch: expected [{}x{}x{}x{}], got {:?}",
                batch_size, out_channels, out_height, out_width, dst_dims
            )));
        }

        // Verify filter dimensions
        if in_channels != filter_in_channels {
            return Err(VKMLEngineError::ShapeMismatch(format!(
                "Filter input channels {} doesn't match input channels {}",
                filter_in_channels, in_channels
            )));
        }

        // Zero initialize result
        for val in dst_data.iter_mut() {
            *val = 0.0;
        }

        // Perform convolution
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0;

                        // Apply kernel
                        for ic in 0..in_channels {
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    // Handle padding correctly
                                    let ih = (oh * self.stride.0 + kh).checked_sub(self.padding.0);
                                    let iw = (ow * self.stride.1 + kw).checked_sub(self.padding.1);

                                    // Only process valid input positions
                                    if let (Some(ih), Some(iw)) = (ih, iw) {
                                        if ih < in_height && iw < in_width {
                                            let in_idx = ((b * in_channels + ic) * in_height + ih)
                                                * in_width
                                                + iw;
                                            let w_idx = ((oc * in_channels + ic) * kernel_h + kh)
                                                * kernel_w
                                                + kw;

                                            sum += src_data[in_idx] * weights_data[w_idx];
                                        }
                                    }
                                }
                            }
                        }

                        // Add bias if present
                        if let Some(bias) = &bias_data {
                            sum += bias[oc];
                        }

                        let out_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                        dst_data[out_idx] = sum;
                    }
                }
            }
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }
}
