use crate::ComputeManager;
use crate::error::VKMLError;
use crate::instruction::gpu_operations::GPUOperation;
use crate::instruction::maxpool::push_constants::{
    MaxPool1DPushConstants, MaxPool2DPushConstants, MaxPool3DPushConstants,
};
use crate::utils::{OnnxAutoPad, as_bytes, calc_begin_and_end_pads};
use crate::{
    gpu::vk_gpu::Gpu,
    instruction::{instruction::Instruction, maxpool::f32_f32_cpu::f32_f32_cpu},
    tensor::TensorDesc,
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

pub struct MaxPoolInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    pub auto_pad: OnnxAutoPad,
    pub dilations: Vec<i64>,
    pub kernel_shape: Vec<i64>,
    pub pads: Vec<i64>,
    pub strides: Vec<i64>,
    pub ceil_mode: bool,
}

impl MaxPoolInstruction {
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

impl Debug for MaxPoolInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "MaxPool(src={}, dst={}, kernel={:?}, strides={:?}, pads={:?}, dilations={:?}, auto_pad={:?}, ceil_mode={})",
            self.src,
            self.dst,
            self.kernel_shape,
            self.strides,
            self.pads,
            self.dilations,
            self.auto_pad,
            self.ceil_mode
        )
    }
}

impl Instruction for MaxPoolInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.src]
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if !new_inputs.is_empty() {
            self.src = new_inputs[0];
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
        // GPU implementation: bind src(0) and dst(2), push constants and dispatch.
        let src_tensor = cm.tensor_read(self.src);
        let dst_tensor = cm.tensor_read(self.dst);

        let src_mem = src_tensor.get_gpu_memory_or_panic();
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        let src_desc = src_tensor.desc();
        let binding_count = 2; // src, dst
        gpu.bind_storage_buffers(command_buffer, &[src_mem, dst_mem]);

        // choose shader based on spatial rank
        let spatial_rank = if src_desc.ndim() >= 2 {
            src_desc.ndim() - 2
        } else {
            0
        };

        let pb = self.compute_pads(src_desc);

        match spatial_rank {
            0 | 1 => {
                let src_dims = src_desc.dims();
                let input_len = if src_dims.len() >= 3 {
                    src_dims[2] as u32
                } else {
                    1
                };
                let dst_desc = dst_tensor.desc();
                let dst_dims = dst_desc.dims();
                let output_len = if dst_dims.len() >= 3 {
                    dst_dims[2] as u32
                } else {
                    1
                };

                let pc = MaxPool1DPushConstants {
                    n: src_dims[0] as u32,
                    c: src_dims[1] as u32,
                    input_len,
                    output_len,
                    kernel: self.kernel_shape.first().copied().unwrap_or(1) as u32,
                    stride: self.strides.first().copied().unwrap_or(1) as u32,
                    dilation: self.dilations.first().copied().unwrap_or(1) as u32,
                    pad_begin: pb.first().copied().unwrap_or(0) as u32,
                };

                let push_constant_bytes = as_bytes(&pc);

                // choose local workgroup size and bind specialized pipeline
                let total: u64 = (src_dims[0] as u64) * (src_dims[1] as u64) * (output_len as u64);
                let local_size = gpu.optimal_workgroup_size_1d(total);

                let src_dtype = src_desc.data_type();
                let dst_dtype = dst_desc.data_type();
                let gpu_op = match (src_dtype, dst_dtype) {
                    (DataType::Float, DataType::Float) => GPUOperation::MaxPool1D_F32_F32,
                    _ => {
                        return Err(VKMLError::Instruction(format!(
                            "GPU MaxPool unimplemented for DataType src:{:?}, dst:{:?}",
                            src_dtype, dst_dtype
                        )));
                    }
                };

                gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);
                gpu.bind_push_constants(command_buffer, binding_count, push_constant_bytes);

                gpu.dispatch(command_buffer, local_size, [total, 1, 1]);
            }
            2 => {
                let src_dims = src_desc.dims();
                let dst_desc = dst_tensor.desc();
                let dst_dims = dst_desc.dims();

                let pc = MaxPool2DPushConstants {
                    n: src_dims[0] as u32,
                    c: src_dims[1] as u32,
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
                };

                let push_constant_bytes = as_bytes(&pc);

                // choose local tile size and bind specialized pipeline
                let out_w = dst_dims[3] as u64;
                let out_h = dst_dims[2] as u64;
                let batch_nc = (dst_dims[0] as u64) * (dst_dims[1] as u64); // n * c

                let local_size = gpu.optimal_workgroup_size_2d(out_h, out_w);

                let src_dtype = src_desc.data_type();
                let dst_dtype = dst_desc.data_type();
                let gpu_op = match (src_dtype, dst_dtype) {
                    (DataType::Float, DataType::Float) => GPUOperation::MaxPool2D_F32_F32,
                    (DataType::Float16, DataType::Float16) => GPUOperation::MaxPool2D_F16_F16,
                    _ => {
                        return Err(VKMLError::Instruction(format!(
                            "GPU MaxPool unimplemented for DataType src:{:?}, dst:{:?}",
                            src_dtype, dst_dtype
                        )));
                    }
                };

                gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);
                gpu.bind_push_constants(command_buffer, binding_count, push_constant_bytes);

                gpu.dispatch(command_buffer, local_size, [out_w, out_h, batch_nc]);
            }
            3 => {
                let src_dims = src_desc.dims();
                let dst_desc = dst_tensor.desc();
                let dst_dims = dst_desc.dims();

                let pc = MaxPool3DPushConstants {
                    n: src_dims[0] as u32,
                    c: src_dims[1] as u32,
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
                };

                let push_constant_bytes = as_bytes(&pc);

                let out_w = dst_dims[4] as u64;
                let out_h = dst_dims[3] as u64;
                let out_d = dst_dims[2] as u64;

                let total_z = out_d * (dst_dims[0] as u64) * (dst_dims[1] as u64);

                let local_size = gpu.optimal_workgroup_size_3d(out_w, out_h, out_d);

                let src_dtype = src_desc.data_type();
                let dst_dtype = dst_desc.data_type();
                let gpu_op = match (src_dtype, dst_dtype) {
                    (DataType::Float, DataType::Float) => GPUOperation::MaxPool3D_F32_F32,
                    _ => {
                        return Err(VKMLError::Instruction(format!(
                            "GPU MaxPool unimplemented for DataType src:{:?}, dst:{:?}",
                            src_dtype, dst_dtype
                        )));
                    }
                };

                gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);
                gpu.bind_push_constants(command_buffer, binding_count, push_constant_bytes);

                gpu.dispatch(command_buffer, local_size, [out_w, out_h, total_z]);
            }
            _ => panic!("Unsupported spatial rank {} for GPU MaxPool", spatial_rank),
        }

        Ok(())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        // Acquire read guard and dst write guard
        let src_guard = cm.tensor_read(self.src);
        let src_desc = src_guard.desc();
        let src_bytes = src_guard.get_cpu_memory_slice_or_panic();

        let dst_guard = cm.tensor_write(self.dst);
        let dst_desc = dst_guard.desc().clone();
        let dst_ptr = dst_guard.get_cpu_memory_mut_slice_or_panic();

        let pads_begin = self.compute_pads(src_desc);

        let src_dtype = src_desc.data_type();
        let dst_dtype = dst_desc.data_type();
        match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => {
                f32_f32_cpu(
                    src_desc.dims(),
                    dst_desc.dims(),
                    src_bytes,
                    dst_ptr,
                    &self.kernel_shape,
                    &self.strides,
                    &pads_begin,
                    &self.dilations,
                );
            }
            _ => unimplemented!(
                "MaxPool: unimplemented CPU for DataType src:{:?}, dst:{:?}",
                src_dtype,
                dst_dtype
            ),
        }
    }
}
