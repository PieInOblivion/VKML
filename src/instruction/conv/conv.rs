use crate::{
    gpu::vk_gpu::GPU,
    instruction::{
        conv::f32_cpu::f32_cpu, gpu_operations::GPUMemoryOperation, instruction::Instruction,
    },
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::{vk, vk::DeviceV1_0};

#[derive(Clone)]
pub struct ConvInstruction {
    pub src: TensorId,
    pub weights: TensorId,
    pub bias: Option<TensorId>,
    pub dst: TensorId,

    pub auto_pad: AutoPad,
    pub dilations: Vec<usize>,
    pub group: usize,
    pub kernel_shape: Vec<usize>,
    pub pads: Vec<usize>,
    pub strides: Vec<usize>,
}

/// How to compute padding for convolution when unspecified
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AutoPad {
    NotSet,
    Valid,
    SameUpper,
    SameLower,
}

impl Debug for ConvInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Conv(src={}, weights={}, bias={:?}, dst={}, auto_pad={:?}, dilations={:?}, group={:?}, kernel_shape={:?}, pads={:?}, strides={:?})",
            self.src,
            self.weights,
            self.bias,
            self.dst,
            self.auto_pad,
            self.dilations,
            self.group,
            self.kernel_shape,
            self.pads,
            self.strides
        )
    }
}

impl Instruction for ConvInstruction {
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
        // Acquire read guards for tensors so we can access descriptors and GPU memory
        let src_tensor = tensor_graph.tensor_read(self.src);
        let weights_tensor = tensor_graph.tensor_read(self.weights);
        let dst_tensor = tensor_graph.tensor_read(self.dst);

        let src_mem = src_tensor.get_gpu_memory_or_panic();
        let weights_mem = weights_tensor.get_gpu_memory_or_panic();
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        // Optional bias read guard (kept in scope)
        let bias_tensor_opt = if let Some(bid) = self.bias {
            Some(tensor_graph.tensor_read(bid))
        } else {
            None
        };

        let bias_mem = bias_tensor_opt
            .as_ref()
            .map(|t| t.get_gpu_memory_or_panic());

        Ok(())
    }

    fn execute_cpu(&self, tensor_graph: &TensorGraph) {
        // Acquire read guards and extract copies of metadata and input bytes so we can
        // drop the read guards before taking a mutable write guard on dst.
        let src_guard = tensor_graph.tensor_read(self.src);
        let weights_guard = tensor_graph.tensor_read(self.weights);
        let bias_guard_opt = if let Some(bid) = self.bias {
            Some(tensor_graph.tensor_read(bid))
        } else {
            None
        };

        // Clone descriptors (cheap) and copy input bytes (so we can release read locks)
        let src_desc = src_guard.desc.clone();
        let weight_desc = weights_guard.desc.clone();
        let src_bytes_vec: Vec<u8> = src_guard.get_cpu_memory_slice_or_panic().to_vec();
        let weight_bytes_vec: Vec<u8> = weights_guard.get_cpu_memory_slice_or_panic().to_vec();
        let bias_bytes_vec_opt: Option<Vec<u8>> = bias_guard_opt
            .as_ref()
            .map(|t| t.get_cpu_memory_slice_or_panic().to_vec());

        // drop read guards now
        drop(src_guard);
        drop(weights_guard);
        drop(bias_guard_opt);

        // Obtain dst as mutable write guard
        let mut dst_tensor = tensor_graph.tensor_write(self.dst);
        let dst_desc = dst_tensor.desc.clone();

        // Convert dims to usize vectors
        let src_dims: Vec<usize> = src_desc.to_dims().iter().map(|d| *d as usize).collect();
        let weight_dims: Vec<usize> = weight_desc.to_dims().iter().map(|d| *d as usize).collect();
        let dst_dims: Vec<usize> = dst_desc.to_dims().iter().map(|d| *d as usize).collect();

        // Get raw bytes as slices referencing our copied vecs
        let src_bytes: &[u8] = src_bytes_vec.as_slice();
        let weight_bytes: &[u8] = weight_bytes_vec.as_slice();
        let bias_bytes_opt: Option<&[u8]> = bias_bytes_vec_opt.as_ref().map(|v| v.as_slice());
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        // Spatial rank and normalize kernel/stride/dilation
        let spatial_rank = if src_desc.ndim() >= 2 {
            src_desc.ndim() - 2
        } else {
            0
        };
        let mut stride_vec: Vec<usize> = Vec::with_capacity(spatial_rank);
        let mut dilation_vec: Vec<usize> = Vec::with_capacity(spatial_rank);
        let mut kernel_vec: Vec<usize> = Vec::with_capacity(spatial_rank);
        for i in 0..spatial_rank {
            stride_vec.push(self.strides.get(i).copied().unwrap_or(1));
            dilation_vec.push(self.dilations.get(i).copied().unwrap_or(1));
            kernel_vec.push(self.kernel_shape.get(i).copied().unwrap_or(1));
        }

        // Compute pads_begin and pads_end based on self.pads or auto_pad
        // Note: we compute pads_end to preserve ONNX semantics, for shape
        // inference, validation, and possible pre-padding optimizations.
        // The inner convolution kernel only needs pads_begin for indexing.
        let mut pads_begin: Vec<usize> = vec![0; spatial_rank];
        let mut pads_end: Vec<usize> = vec![0; spatial_rank];

        if self.pads.len() >= spatial_rank * 2 {
            // ONNX style [b1..bn, e1..en]
            for i in 0..spatial_rank {
                pads_begin[i] = self.pads[i];
                pads_end[i] = self.pads[spatial_rank + i];
            }
        } else if self.pads.len() == spatial_rank {
            // symmetric
            for i in 0..spatial_rank {
                pads_begin[i] = self.pads[i];
                pads_end[i] = self.pads[i];
            }
        } else if self.auto_pad != AutoPad::NotSet {
            // Compute SAME_UPPER / SAME_LOWER / VALID pads following ONNX semantics
            // Need input spatial sizes
            let input_spatial: Vec<i64> = src_desc.to_dims()[2..].to_vec();
            for i in 0..spatial_rank {
                let in_i = input_spatial[i] as i64;
                let k = kernel_vec[i] as i64;
                let s = stride_vec[i] as i64;
                let d = dilation_vec[i] as i64;

                if self.auto_pad == AutoPad::Valid {
                    pads_begin[i] = 0;
                    pads_end[i] = 0;
                } else {
                    // SAME_UPPER or SAME_LOWER
                    let out = (in_i + s - 1) / s; // ceil
                    let pad_needed = ((out - 1) * s + d * (k - 1) + 1) - in_i;
                    let pad_needed = if pad_needed > 0 { pad_needed } else { 0 } as usize;
                    if self.auto_pad == AutoPad::SameUpper {
                        pads_begin[i] = pad_needed / 2;
                        pads_end[i] = pad_needed - pads_begin[i];
                    } else {
                        // SameLower
                        pads_end[i] = pad_needed / 2;
                        pads_begin[i] = pad_needed - pads_end[i];
                    }
                }
            }
        }

        // Dispatch based on data type
        let dst_data_type = dst_desc.data_type();
        match dst_data_type {
            DataType::Float => {
                f32_cpu(
                    src_dims,
                    weight_dims,
                    dst_dims,
                    src_bytes,
                    weight_bytes,
                    bias_bytes_opt,
                    dst_ptr,
                    stride_vec,
                    pads_begin,
                    dilation_vec,
                    self.group,
                );
            }
            other => panic!(
                "conv.rs execute_cpu: unimplemented CPU Conv for DataType {:?}",
                other
            ),
        }
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }
}
