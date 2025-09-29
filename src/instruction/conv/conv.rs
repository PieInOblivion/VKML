use crate::ComputeManager;
use crate::instruction::conv::push_constants::{
    Conv1DPushConstants, Conv2DPushConstants, Conv3DPushConstants,
};
use crate::tensor::desc::TensorDesc;
use crate::utils::bytes::as_bytes;
use crate::{
    gpu::vk_gpu::Gpu,
    instruction::{
        conv::f32_cpu::f32_cpu, gpu_operations::GPUMemoryOperation, instruction::Instruction,
    },
    tensor_graph::tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk::Handle;
use vulkanalia::vk::KhrPushDescriptorExtension;
use vulkanalia::{vk, vk::DeviceV1_0};

#[derive(Clone)]
pub struct ConvInstruction {
    pub src: TensorId,
    pub weights: TensorId,
    pub bias: Option<TensorId>,
    pub dst: TensorId,

    pub auto_pad: AutoPad,
    pub dilations: Vec<usize>,
    pub group: i64,
    pub kernel_shape: Vec<usize>,
    pub pads: Vec<usize>,
    pub strides: Vec<usize>,
}

impl ConvInstruction {
    // Helper to compute pads_begin and pads_end following the same logic
    // used in execute_cpu (handles explicit pads, symmetric pads, and auto_pad).
    pub fn compute_pads(&self, src_desc: &TensorDesc) -> (Vec<usize>, Vec<usize>) {
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

        let mut pads_begin: Vec<usize> = vec![0; spatial_rank];
        let mut pads_end: Vec<usize> = vec![0; spatial_rank];

        if self.pads.len() >= spatial_rank * 2 {
            for i in 0..spatial_rank {
                pads_begin[i] = self.pads[i];
                pads_end[i] = self.pads[spatial_rank + i];
            }
        } else if self.pads.len() == spatial_rank {
            for i in 0..spatial_rank {
                pads_begin[i] = self.pads[i];
                pads_end[i] = self.pads[i];
            }
        } else if self.auto_pad != AutoPad::NotSet {
            let input_spatial: Vec<i64> = src_desc.to_dims()[2..].to_vec();
            for i in 0..spatial_rank {
                let in_i = input_spatial[i];
                let k = kernel_vec[i] as i64;
                let s = stride_vec[i] as i64;
                let d = dilation_vec[i] as i64;

                if self.auto_pad == AutoPad::Valid {
                    pads_begin[i] = 0;
                    pads_end[i] = 0;
                } else {
                    let out = (in_i + s - 1) / s; // ceil
                    let pad_needed = ((out - 1) * s + d * (k - 1) + 1) - in_i;
                    let pad_needed = if pad_needed > 0 { pad_needed } else { 0 } as usize;
                    if self.auto_pad == AutoPad::SameUpper {
                        pads_begin[i] = pad_needed / 2;
                        pads_end[i] = pad_needed - pads_begin[i];
                    } else {
                        pads_end[i] = pad_needed / 2;
                        pads_begin[i] = pad_needed - pads_end[i];
                    }
                }
            }
        }

        (pads_begin, pads_end)
    }
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
        gpu: &Gpu,
        command_buffer: vk::CommandBuffer,
        cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Acquire read guards for tensors so we can access descriptors and GPU memory
        let src_tensor = cm.tensor_read(self.src);
        let weights_tensor = cm.tensor_read(self.weights);
        let dst_tensor = cm.tensor_read(self.dst);

        // Basic sanity checks for group before doing GPU work
        let src_desc_tmp = src_tensor.desc.clone();
        let c_val = src_desc_tmp.to_dims()[1];
        let dst_desc_tmp = dst_tensor.desc.clone();
        let m_val = dst_desc_tmp.to_dims()[1];
        if self.group < 1 || c_val % self.group != 0 || m_val % self.group != 0 {
            panic!(
                "ConvInstruction.create_command_buffer: invalid group configuration: group={}, C={}, M={}",
                self.group, c_val, m_val
            );
        }

        let src_mem = src_tensor.get_gpu_memory_or_panic();
        let weights_mem = weights_tensor.get_gpu_memory_or_panic();
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        // Optional bias read guard (kept in scope)
        let bias_tensor_opt = self.bias.map(|bid| cm.tensor_read(bid));

        let bias_mem = bias_tensor_opt
            .as_ref()
            .map(|t| t.get_gpu_memory_or_panic());

        // Prepare push constants and descriptor set
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };

            gpu.get_device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            // Prepare buffer infos for bindings: src(0), weights(1), dst(2), bias(3)
            let buffer_infos = [
                vk::DescriptorBufferInfo {
                    buffer: src_mem.buffer,
                    offset: 0,
                    range: src_mem.size,
                },
                vk::DescriptorBufferInfo {
                    buffer: weights_mem.buffer,
                    offset: 0,
                    range: weights_mem.size,
                },
                vk::DescriptorBufferInfo {
                    buffer: dst_mem.buffer,
                    offset: 0,
                    range: dst_mem.size,
                },
                vk::DescriptorBufferInfo {
                    buffer: if let Some(b) = bias_mem {
                        b.buffer
                    } else {
                        vk::Buffer::null()
                    },
                    offset: 0,
                    range: if let Some(b) = bias_mem { b.size } else { 0 },
                },
            ];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: vk::DescriptorSet::null(),
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[0],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: vk::DescriptorSet::null(),
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[1],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: vk::DescriptorSet::null(),
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[2],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: vk::DescriptorSet::null(),
                    dst_binding: 3,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[3],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
            ];

            // Decide which shader/pipeline to use based on spatial rank
            let src_desc = &src_tensor.desc;
            let spatial_rank = if src_desc.ndim() >= 2 {
                src_desc.ndim() - 2
            } else {
                0
            };

            // Prepare push-constant bytes per shader
            match spatial_rank {
                0 | 1 => {
                    // 1D shader
                    // derive dims
                    let src_dims = src_desc.to_dims();
                    let input_len = if src_dims.len() >= 3 {
                        src_dims[2] as u32
                    } else {
                        1
                    };

                    let dst_desc = &dst_tensor.desc;
                    let dst_dims = dst_desc.to_dims();
                    let output_len = if dst_dims.len() >= 3 {
                        dst_dims[2] as u32
                    } else {
                        1
                    };

                    let pc_values = Conv1DPushConstants {
                        n: src_dims[0] as u32,
                        c: src_dims[1] as u32,
                        m: dst_dims[1] as u32,
                        input_len,
                        output_len,
                        kernel: self.kernel_shape.first().copied().unwrap_or(1) as u32,
                        stride: self.strides.first().copied().unwrap_or(1) as u32,
                        dilation: self.dilations.first().copied().unwrap_or(1) as u32,
                        // compute pads using the same helper as CPU path
                        pad_begin: self.compute_pads(src_desc).0.first().copied().unwrap_or(0)
                            as u32,
                        group: self.group as u32,
                        has_bias: if self.bias.is_some() { 1 } else { 0 },
                    };

                    let push_constant_bytes = as_bytes(&pc_values);

                    let pipeline = gpu.get_or_create_pipeline(GPUMemoryOperation::Conv1D_F32);

                    gpu.get_device().cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline,
                    );
                    gpu.get_device().cmd_push_descriptor_set_khr(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        gpu.get_layout(),
                        0,
                        &write_descriptor_sets,
                    );

                    gpu.get_device().cmd_push_constants(
                        command_buffer,
                        gpu.get_layout(),
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        push_constant_bytes,
                    );

                    // dispatch: output elements = n * m * output_len
                    let total: u64 =
                        (src_dims[0] as u64) * (dst_dims[1] as u64) * (output_len as u64);
                    let workgroup_size = 256u64;
                    let num_groups = total.div_ceil(workgroup_size) as u32;
                    gpu.get_device()
                        .cmd_dispatch(command_buffer, num_groups, 1, 1);
                }
                2 => {
                    // 2D shader
                    let src_dims = src_desc.to_dims();
                    let dst_desc = &dst_tensor.desc;
                    let dst_dims = dst_desc.to_dims();

                    let pc_values = Conv2DPushConstants {
                        n: src_dims[0] as u32,
                        c: src_dims[1] as u32,
                        m: dst_dims[1] as u32,
                        in_h: src_dims[2] as u32,
                        in_w: src_dims[3] as u32,
                        out_h: dst_dims[2] as u32,
                        out_w: dst_dims[3] as u32,
                        k_h: self.kernel_shape.first().copied().unwrap_or(1) as u32,
                        k_w: self.kernel_shape.get(1).copied().unwrap_or(1) as u32,
                        s_h: self.strides.first().copied().unwrap_or(1) as u32,
                        s_w: self.strides.get(1).copied().unwrap_or(1) as u32,
                        d_h: self.dilations.first().copied().unwrap_or(1) as u32,
                        d_w: self.dilations.get(1).copied().unwrap_or(1) as u32,
                        // compute pads using the same helper as CPU path
                        pad_h: self.compute_pads(src_desc).0.first().copied().unwrap_or(0) as u32,
                        pad_w: self.compute_pads(src_desc).0.get(1).copied().unwrap_or(0) as u32,
                        group: self.group as u32,
                        has_bias: if self.bias.is_some() { 1 } else { 0 },
                    };

                    let push_constant_bytes = as_bytes(&pc_values);

                    let pipeline = gpu.get_or_create_pipeline(GPUMemoryOperation::Conv2D_F32);

                    gpu.get_device().cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline,
                    );

                    gpu.get_device().cmd_push_descriptor_set_khr(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        gpu.get_layout(),
                        0,
                        &write_descriptor_sets,
                    );

                    gpu.get_device().cmd_push_constants(
                        command_buffer,
                        gpu.get_layout(),
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        push_constant_bytes,
                    );

                    // dispatch: out_w x out_h workgroups, z dimension encodes (m * n)
                    let out_w = dst_dims[3] as u32;
                    let out_h = dst_dims[2] as u32;
                    let groups_x = out_w.div_ceil(16);
                    let groups_y = out_h.div_ceil(16);
                    let groups_z = (dst_dims[0] as u32) * (dst_dims[1] as u32); // n * m

                    gpu.get_device()
                        .cmd_dispatch(command_buffer, groups_x, groups_y, groups_z);
                }
                3 => {
                    // 3D shader
                    let src_dims = src_desc.to_dims();
                    let dst_desc = &dst_tensor.desc;
                    let dst_dims = dst_desc.to_dims();

                    let pc_values = Conv3DPushConstants {
                        n: src_dims[0] as u32,
                        c: src_dims[1] as u32,
                        m: dst_dims[1] as u32,
                        in_d: src_dims[2] as u32,
                        in_h: src_dims[3] as u32,
                        in_w: src_dims[4] as u32,
                        out_d: dst_dims[2] as u32,
                        out_h: dst_dims[3] as u32,
                        out_w: dst_dims[4] as u32,
                        k_d: self.kernel_shape.first().copied().unwrap_or(1) as u32,
                        k_h: self.kernel_shape.get(1).copied().unwrap_or(1) as u32,
                        k_w: self.kernel_shape.get(2).copied().unwrap_or(1) as u32,
                        s_d: self.strides.first().copied().unwrap_or(1) as u32,
                        s_h: self.strides.get(1).copied().unwrap_or(1) as u32,
                        s_w: self.strides.get(2).copied().unwrap_or(1) as u32,
                        d_d: self.dilations.first().copied().unwrap_or(1) as u32,
                        d_h: self.dilations.get(1).copied().unwrap_or(1) as u32,
                        d_w: self.dilations.get(2).copied().unwrap_or(1) as u32,
                        pad_d: self.compute_pads(src_desc).0.first().copied().unwrap_or(0) as u32,
                        pad_h: self.compute_pads(src_desc).0.get(1).copied().unwrap_or(0) as u32,
                        pad_w: self.compute_pads(src_desc).0.get(2).copied().unwrap_or(0) as u32,
                        group: {
                            if self.group < 1 {
                                panic!(
                                    "ConvInstruction.create_command_buffer: group must be >= 1, got {}",
                                    self.group
                                );
                            }
                            self.group as u32
                        },
                        has_bias: if self.bias.is_some() { 1 } else { 0 },
                    };

                    let push_constant_bytes = as_bytes(&pc_values);

                    let pipeline = gpu.get_or_create_pipeline(GPUMemoryOperation::Conv3D_F32);

                    gpu.get_device().cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline,
                    );

                    gpu.get_device().cmd_push_descriptor_set_khr(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        gpu.get_layout(),
                        0,
                        &write_descriptor_sets,
                    );

                    gpu.get_device().cmd_push_constants(
                        command_buffer,
                        gpu.get_layout(),
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        push_constant_bytes,
                    );

                    // dispatch: x=ceil(out_w/8), y=ceil(out_h/8), z=ceil((out_d * n * m) / local_z)
                    let groups_x = (dst_dims[4] as u32).div_ceil(8);
                    let groups_y = (dst_dims[3] as u32).div_ceil(8);
                    let local_z = 4u32; // shader local size z
                    let total_z =
                        (dst_dims[2] as u32) * (dst_dims[0] as u32) * (dst_dims[1] as u32);
                    let groups_z = total_z.div_ceil(local_z);

                    gpu.get_device()
                        .cmd_dispatch(command_buffer, groups_x, groups_y, groups_z);
                }
                _ => panic!("Unsupported spatial rank {} for GPU conv", spatial_rank),
            }

            gpu.get_device().end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        // Acquire read guards and extract copies of metadata and input bytes so we can
        // drop the read guards before taking a mutable write guard on dst.
        let src_guard = cm.tensor_read(self.src);
        let weights_guard = cm.tensor_read(self.weights);
        let bias_guard_opt = self.bias.map(|bid| cm.tensor_read(bid));

        // Clone descriptors (cheap) and copy input bytes (so we can release read locks)
        let src_desc = src_guard.desc.clone();
        let weight_desc = weights_guard.desc.clone();
        let src_bytes_vec: Vec<u8> = src_guard.get_cpu_memory_slice_or_panic().to_vec();
        let weight_bytes_vec: Vec<u8> = weights_guard.get_cpu_memory_slice_or_panic().to_vec();
        let bias_bytes_vec_opt: Option<Vec<u8>> = bias_guard_opt
            .as_ref()
            .map(|t| t.get_cpu_memory_slice_or_panic().to_vec());

        // Obtain dst as mutable write guard
        let dst_tensor = cm.tensor_write(self.dst);
        let dst_desc = dst_tensor.desc.clone();

        // Convert dims to usize vectors
        let src_dims: Vec<usize> = src_desc.to_dims().iter().map(|d| *d as usize).collect();
        let weight_dims: Vec<usize> = weight_desc.to_dims().iter().map(|d| *d as usize).collect();
        let dst_dims: Vec<usize> = dst_desc.to_dims().iter().map(|d| *d as usize).collect();

        // Get raw bytes as slices referencing our copied vecs
        let src_bytes: &[u8] = src_bytes_vec.as_slice();
        let weight_bytes: &[u8] = weight_bytes_vec.as_slice();
        let bias_bytes_opt: Option<&[u8]> = bias_bytes_vec_opt.as_deref();
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
                let in_i = input_spatial[i];
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
                    self.group as usize,
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
