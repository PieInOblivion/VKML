use crate::ComputeManager;
use crate::error::VKMLError;
use crate::instruction::conv::push_constants::{
    Conv1DPushConstants, Conv2DPushConstants, Conv3DPushConstants,
};
use crate::tensor::TensorDesc;
use crate::utils::bytes::as_bytes;
use crate::utils::{OnnxAutoPad, calc_begin_and_end_pads};
use crate::{
    gpu::vk_gpu::Gpu,
    instruction::{
        conv::f32_f32_f32_f32_cpu::f32_f32_f32_f32_cpu, gpu_operations::GPUOperation,
        instruction::Instruction,
    },
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

pub struct ConvInstruction {
    pub src: TensorId,
    pub weights: TensorId,
    pub bias: Option<TensorId>,
    pub dst: TensorId,

    pub auto_pad: OnnxAutoPad,
    pub dilations: Vec<i64>,
    pub group: i64,
    pub kernel_shape: Vec<i64>,
    pub pads: Vec<i64>,
    pub strides: Vec<i64>,
}

impl ConvInstruction {
    fn compute_pads(&self, src_desc: &TensorDesc) -> Vec<i64> {
        let (pb, _pe) = calc_begin_and_end_pads(
            self.auto_pad.clone(),
            &self.pads,
            &self.kernel_shape,
            &self.strides,
            &self.dilations,
            src_desc,
        );
        pb
    }
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

    fn record_into_command_buffer(
        &self,
        gpu: &Gpu,
        command_buffer: vk::CommandBuffer,
        cm: &ComputeManager,
    ) -> Result<(), VKMLError> {
        // Acquire read guards for tensors so we can access descriptors and GPU memory
        let src_tensor = cm.tensor_read(self.src);
        let weights_tensor = cm.tensor_read(self.weights);
        let dst_tensor = cm.tensor_read(self.dst);

        // Basic sanity checks for group before doing GPU work
        let src_desc_tmp = src_tensor.desc.clone();
        let c_val = src_desc_tmp.dims()[1];
        let dst_desc_tmp = dst_tensor.desc.clone();
        let m_val = dst_desc_tmp.dims()[1];
        if self.group < 1 || c_val % self.group != 0 || m_val % self.group != 0 {
            panic!(
                "ConvInstruction.record_into_command_buffer: invalid group configuration: group={}, C={}, M={}",
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

        // Decide which shader/pipeline to use based on spatial rank and prepare push constants
        let src_desc = &src_tensor.desc;
        let spatial_rank = if src_desc.ndim() >= 2 {
            src_desc.ndim() - 2
        } else {
            0
        };

        let pb = self.compute_pads(src_desc);

        match spatial_rank {
            0 | 1 => {
                // 1D shader
                let src_dims = src_desc.dims();
                let input_len = if src_dims.len() >= 3 {
                    src_dims[2] as u32
                } else {
                    1
                };
                let dst_desc = &dst_tensor.desc;
                let dst_dims = dst_desc.dims();
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
                    pad_begin: pb.first().copied().unwrap_or(0) as u32,
                    group: self.group as u32,
                    has_bias: if self.bias.is_some() { 1 } else { 0 },
                };

                let push_constant_bytes = as_bytes(&pc_values);

                // Bind pipeline and descriptors (preserve optional bias binding)
                // choose an optimal local workgroup size for this 1D workload
                let total: u64 = (src_dims[0] as u64) * (dst_dims[1] as u64) * (output_len as u64);
                let local_size = gpu.optimal_workgroup_size_1d(total);

                let src_dtype = src_desc.data_type();
                let weight_dtype = weights_tensor.desc.data_type();
                let bias_dtype_opt = bias_tensor_opt.as_ref().map(|t| t.desc.data_type());
                let dst_dtype = dst_desc.data_type();
                let gpu_op = match (src_dtype, weight_dtype, bias_dtype_opt, dst_dtype) {
                    (DataType::Float, DataType::Float, None, DataType::Float)
                    | (DataType::Float, DataType::Float, Some(DataType::Float), DataType::Float) => {
                        GPUOperation::Conv1D_F32_F32_F32_F32
                    }
                    _ => {
                        return Err(VKMLError::Instruction(format!(
                            "GPU Conv unimplemented for DataType src:{:?}, weight:{:?}, bias:{:?}, dst:{:?}",
                            src_dtype,
                            weight_dtype,
                            bias_dtype_opt
                                .map(|dt| format!("{:?}", dt))
                                .unwrap_or_else(|| "None".to_string()),
                            dst_dtype
                        )));
                    }
                };

                let binding_count = 4; // src, weights, dst, bias (optional)

                gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);
                gpu.bind_storage_buffers_optional(
                    command_buffer,
                    &[Some(src_mem), Some(weights_mem), Some(dst_mem), bias_mem],
                );

                gpu.bind_push_constants(command_buffer, binding_count, push_constant_bytes);

                // dispatch: provide total work counts per-dimension; Gpu::dispatch will
                // compute the needed number of workgroups as ceil(work/local_size)
                gpu.dispatch(command_buffer, local_size, [total, 1, 1]);
            }
            2 => {
                // 2D shader
                let src_dims = src_desc.dims();
                let dst_desc = &dst_tensor.desc;
                let dst_dims = dst_desc.dims();

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
                    pad_h: pb.first().copied().unwrap_or(0) as u32,
                    pad_w: pb.get(1).copied().unwrap_or(0) as u32,
                    group: self.group as u32,
                    has_bias: if self.bias.is_some() { 1 } else { 0 },
                };

                let push_constant_bytes = as_bytes(&pc_values);

                // choose a 2D tile size suitable for (out_h x out_w) work
                let out_w = dst_dims[3] as u64;
                let out_h = dst_dims[2] as u64;
                let batch_nm = (dst_dims[0] as u64) * (dst_dims[1] as u64); // n * m

                let local_size = gpu.optimal_workgroup_size_2d(out_h, out_w);

                let src_dtype = src_desc.data_type();
                let weight_dtype = weights_tensor.desc.data_type();
                let bias_dtype_opt = bias_tensor_opt.as_ref().map(|t| t.desc.data_type());
                let dst_dtype = dst_desc.data_type();
                let gpu_op = match (src_dtype, weight_dtype, bias_dtype_opt, dst_dtype) {
                    (DataType::Float, DataType::Float, None, DataType::Float)
                    | (DataType::Float, DataType::Float, Some(DataType::Float), DataType::Float) => {
                        GPUOperation::Conv2D_F32_F32_F32_F32
                    }
                    (DataType::Float16, DataType::Float16, None, DataType::Float16)
                    | (
                        DataType::Float16,
                        DataType::Float16,
                        Some(DataType::Float16),
                        DataType::Float16,
                    ) => GPUOperation::Conv2D_F16_F16_F16_F16,
                    _ => {
                        return Err(VKMLError::Instruction(format!(
                            "GPU Conv unimplemented for DataType src:{:?}, weight:{:?}, bias:{:?}, dst:{:?}",
                            src_dtype,
                            weight_dtype,
                            bias_dtype_opt
                                .map(|dt| format!("{:?}", dt))
                                .unwrap_or_else(|| "None".to_string()),
                            dst_dtype
                        )));
                    }
                };

                let binding_count = 4; // src, weights, dst, bias (optional)

                gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);
                gpu.bind_storage_buffers_optional(
                    command_buffer,
                    &[Some(src_mem), Some(weights_mem), Some(dst_mem), bias_mem],
                );

                gpu.bind_push_constants(command_buffer, binding_count, push_constant_bytes);

                // dispatch using total work extents (width, height, batch)
                gpu.dispatch(command_buffer, local_size, [out_w, out_h, batch_nm]);
            }
            3 => {
                // 3D shader
                let src_dims = src_desc.dims();
                let dst_desc = &dst_tensor.desc;
                let dst_dims = dst_desc.dims();

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
                    pad_d: pb.first().copied().unwrap_or(0) as u32,
                    pad_h: pb.get(1).copied().unwrap_or(0) as u32,
                    pad_w: pb.get(2).copied().unwrap_or(0) as u32,
                    group: {
                        if self.group < 1 {
                            panic!(
                                "ConvInstruction.record_into_command_buffer: group must be >= 1, got {}",
                                self.group
                            );
                        }
                        self.group as u32
                    },
                    has_bias: if self.bias.is_some() { 1 } else { 0 },
                };

                let push_constant_bytes = as_bytes(&pc_values);

                let out_w = dst_dims[4] as u64;
                let out_h = dst_dims[3] as u64;
                let out_d = dst_dims[2] as u64;

                // total_z includes depth * batch (n * m)
                let total_z = (out_d) * (dst_dims[0] as u64) * (dst_dims[1] as u64);

                // pick a cubic local workgroup size based on spatial dims
                let local_size = gpu.optimal_workgroup_size_3d(out_w, out_h, out_d);

                let src_dtype = src_desc.data_type();
                let weight_dtype = weights_tensor.desc.data_type();
                let bias_dtype_opt = bias_tensor_opt.as_ref().map(|t| t.desc.data_type());
                let dst_dtype = dst_desc.data_type();
                let gpu_op = match (src_dtype, weight_dtype, bias_dtype_opt, dst_dtype) {
                    (DataType::Float, DataType::Float, None, DataType::Float)
                    | (DataType::Float, DataType::Float, Some(DataType::Float), DataType::Float) => {
                        GPUOperation::Conv3D_F32_F32_F32_F32
                    }
                    _ => {
                        return Err(VKMLError::Instruction(format!(
                            "GPU Conv unimplemented for DataType src:{:?}, weight:{:?}, bias:{:?}, dst:{:?}",
                            src_dtype,
                            weight_dtype,
                            bias_dtype_opt
                                .map(|dt| format!("{:?}", dt))
                                .unwrap_or_else(|| "None".to_string()),
                            dst_dtype
                        )));
                    }
                };

                let binding_count = 4; // src, weights, dst, bias (optional)

                gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);
                gpu.bind_storage_buffers_optional(
                    command_buffer,
                    &[Some(src_mem), Some(weights_mem), Some(dst_mem), bias_mem],
                );

                gpu.bind_push_constants(command_buffer, binding_count, push_constant_bytes);

                // dispatch over (w, h, depth * batch)
                gpu.dispatch(command_buffer, local_size, [out_w, out_h, total_z]);
            }
            _ => unimplemented!("Unsupported spatial rank {} for GPU conv", spatial_rank),
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

        // Get raw bytes as slices referencing our copied vecs
        let src_bytes: &[u8] = src_bytes_vec.as_slice();
        let weight_bytes: &[u8] = weight_bytes_vec.as_slice();
        let bias_bytes_opt: Option<&[u8]> = bias_bytes_vec_opt.as_deref();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        let pads_begin = self.compute_pads(&src_desc);

        // Dispatch based on data type
        let src_dtype = src_desc.data_type();
        let weight_dtype = weight_desc.data_type();
        let bias_dtype_opt = bias_guard_opt.as_ref().map(|t| t.desc.data_type());
        let dst_dtype = dst_desc.data_type();
        match (src_dtype, weight_dtype, bias_dtype_opt, dst_dtype) {
            (DataType::Float, DataType::Float, None, DataType::Float)
            | (DataType::Float, DataType::Float, Some(DataType::Float), DataType::Float) => {
                f32_f32_f32_f32_cpu(
                    src_desc.dims(),
                    weight_desc.dims(),
                    dst_desc.dims(),
                    src_bytes,
                    weight_bytes,
                    bias_bytes_opt,
                    dst_ptr,
                    &self.strides,
                    &pads_begin,
                    &self.dilations,
                    self.group as usize,
                );
            }
            _ => unimplemented!(
                "CPU Conv unimplemented for DataType src:{:?}, weight:{:?}, bias:{:?}, dst:{:?}",
                src_dtype,
                weight_dtype,
                bias_dtype_opt
                    .map(|dt| format!("{:?}", dt))
                    .unwrap_or_else(|| "None".to_string()),
                dst_dtype
            ),
        }
    }
}
