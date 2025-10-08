use crate::ComputeManager;
use crate::instruction::AutoPad;
use crate::instruction::gpu_operations::GPUOperation;
use crate::instruction::maxpool::push_constants::{
    MaxPool1DPushConstants, MaxPool2DPushConstants, MaxPool3DPushConstants,
};
use crate::utils::as_bytes;
use crate::{
    gpu::vk_gpu::Gpu,
    instruction::{instruction::Instruction, maxpool::f32_cpu::f32_cpu},
    tensor::desc::TensorDesc,
    tensor_graph::tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

#[derive(Clone)]
pub struct MaxPoolInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    pub auto_pad: AutoPad,
    pub dilations: Vec<usize>,
    pub kernel_shape: Vec<usize>,
    pub pads: Vec<usize>,
    pub strides: Vec<usize>,
    pub ceil_mode: bool,
}

impl MaxPoolInstruction {
    // Helper to compute pads_begin and pads_end following same logic as ConvInstruction
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
            pads_begin[..spatial_rank].copy_from_slice(&self.pads[..spatial_rank]);
            pads_end[..spatial_rank]
                .copy_from_slice(&self.pads[spatial_rank..(spatial_rank + spatial_rank)]);
        } else if self.pads.len() == spatial_rank {
            pads_begin[..spatial_rank].copy_from_slice(&self.pads[..spatial_rank]);
            pads_end[..spatial_rank].copy_from_slice(&self.pads[..spatial_rank]);
        } else if self.auto_pad != AutoPad::NotSet {
            for i in 0..spatial_rank {
                let in_i = src_desc.dims()[i + 2];
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

    fn create_command_buffer(
        &self,
        gpu: &Gpu,
        command_buffer: vk::CommandBuffer,
        cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // GPU implementation: bind src(0) and dst(2), push constants and dispatch.
        let src_tensor = cm.tensor_read(self.src);
        let dst_tensor = cm.tensor_read(self.dst);

        let src_mem = src_tensor.get_gpu_memory_or_panic();
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        let src_desc = &src_tensor.desc;

        gpu.begin_command_buffer(command_buffer)?;

        gpu.bind_storage_buffers(command_buffer, &[&src_mem, &dst_mem]);

        // choose shader based on spatial rank
        let spatial_rank = if src_desc.ndim() >= 2 {
            src_desc.ndim() - 2
        } else {
            0
        };

        match spatial_rank {
            0 | 1 => {
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

                let pc = MaxPool1DPushConstants {
                    n: src_dims[0] as u32,
                    c: src_dims[1] as u32,
                    input_len,
                    output_len,
                    kernel: self.kernel_shape.first().copied().unwrap_or(1) as u32,
                    stride: self.strides.first().copied().unwrap_or(1) as u32,
                    dilation: self.dilations.first().copied().unwrap_or(1) as u32,
                    pad_begin: {
                        let (pb, _pe) = self.compute_pads(src_desc);
                        pb.first().copied().unwrap_or(0) as u32
                    },
                };

                let push_constant_bytes = as_bytes(&pc);

                // choose local workgroup size and bind specialized pipeline
                let total: u64 = (src_dims[0] as u64) * (src_dims[1] as u64) * (output_len as u64);
                let local_size = gpu.optimal_workgroup_size_1d(total);

                gpu.bind_compute_pipeline(command_buffer, GPUOperation::MaxPool1D_F32, local_size);
                gpu.bind_push_constants(command_buffer, push_constant_bytes);

                gpu.dispatch(command_buffer, local_size, [total, 1, 1]);
            }
            2 => {
                let src_dims = src_desc.dims();
                let dst_desc = &dst_tensor.desc;
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
                    pad_h: {
                        let (pb, _pe) = self.compute_pads(src_desc);
                        pb.first().copied().unwrap_or(0) as u32
                    },
                    pad_w: {
                        let (pb, _pe) = self.compute_pads(src_desc);
                        pb.get(1).copied().unwrap_or(0) as u32
                    },
                };

                let push_constant_bytes = as_bytes(&pc);

                // choose local tile size and bind specialized pipeline
                let out_w = dst_dims[3] as u64;
                let out_h = dst_dims[2] as u64;
                let batch_nc = (dst_dims[0] as u64) * (dst_dims[1] as u64); // n * c

                let local_size = gpu.optimal_workgroup_size_2d(out_h, out_w);

                gpu.bind_compute_pipeline(command_buffer, GPUOperation::MaxPool2D_F32, local_size);
                gpu.bind_push_constants(command_buffer, push_constant_bytes);

                gpu.dispatch(command_buffer, local_size, [out_w, out_h, batch_nc]);
            }
            3 => {
                let src_dims = src_desc.dims();
                let dst_desc = &dst_tensor.desc;
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
                    pad_d: {
                        let (pb, _pe) = self.compute_pads(src_desc);
                        pb.first().copied().unwrap_or(0) as u32
                    },
                    pad_h: {
                        let (pb, _pe) = self.compute_pads(src_desc);
                        pb.get(1).copied().unwrap_or(0) as u32
                    },
                    pad_w: {
                        let (pb, _pe) = self.compute_pads(src_desc);
                        pb.get(2).copied().unwrap_or(0) as u32
                    },
                };

                let push_constant_bytes = as_bytes(&pc);

                let out_w = dst_dims[4] as u64;
                let out_h = dst_dims[3] as u64;
                let out_d = dst_dims[2] as u64;

                let total_z = out_d * (dst_dims[0] as u64) * (dst_dims[1] as u64);

                let local_size = gpu.optimal_workgroup_size_3d(out_w, out_h, out_d);

                gpu.bind_compute_pipeline(command_buffer, GPUOperation::MaxPool3D_F32, local_size);
                gpu.bind_push_constants(command_buffer, push_constant_bytes);

                gpu.dispatch(command_buffer, local_size, [out_w, out_h, total_z]);
            }
            _ => panic!("Unsupported spatial rank {} for GPU MaxPool", spatial_rank),
        }

        gpu.end_command_buffer(command_buffer)?;

        Ok(())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        // Acquire read guard and dst write guard
        let src_guard = cm.tensor_read(self.src);
        let src_desc = src_guard.desc.clone();
        let src_bytes = src_guard.get_cpu_memory_slice_or_panic();

        let dst_guard = cm.tensor_write(self.dst);
        let dst_desc = dst_guard.desc.clone();
        let dst_ptr = dst_guard.get_cpu_memory_mut_slice_or_panic();

        // derive dims
        let src_dims: Vec<usize> = src_desc.dims().iter().map(|d| *d as usize).collect();
        let dst_dims: Vec<usize> = dst_desc.dims().iter().map(|d| *d as usize).collect();

        // normalize parameters
        let spatial_rank = if src_dims.len() >= 2 {
            src_dims.len() - 2
        } else {
            0
        };
        let mut stride_vec = vec![1usize; spatial_rank];
        let mut dil_vec = vec![1usize; spatial_rank];
        let mut kernel_vec = vec![1usize; spatial_rank];
        for i in 0..spatial_rank {
            stride_vec[i] = self.strides.get(i).copied().unwrap_or(1);
            dil_vec[i] = self.dilations.get(i).copied().unwrap_or(1);
            kernel_vec[i] = self.kernel_shape.get(i).copied().unwrap_or(1);
        }

        // compute pads_begin according to same rules as conv
        let (pads_begin, _pads_end) = {
            let mut pads_begin = vec![0usize; spatial_rank];
            let mut pads_end = vec![0usize; spatial_rank];

            if self.pads.len() >= spatial_rank * 2 {
                pads_begin[..spatial_rank].copy_from_slice(&self.pads[..spatial_rank]);
                pads_end[..spatial_rank]
                    .copy_from_slice(&self.pads[spatial_rank..(spatial_rank * 2)]);
            } else if self.pads.len() == spatial_rank {
                pads_begin[..spatial_rank].copy_from_slice(&self.pads[..spatial_rank]);
                pads_end[..spatial_rank].copy_from_slice(&self.pads[..spatial_rank]);
            } else if self.auto_pad != AutoPad::NotSet {
                for i in 0..spatial_rank {
                    let in_i = src_desc.dims()[i + 2];
                    let k = kernel_vec[i] as i64;
                    let s = stride_vec[i] as i64;
                    let d = dil_vec[i] as i64;

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
        };

        // call f32_cpu helper
        match dst_desc.data_type() {
            DataType::Float => {
                f32_cpu(
                    src_dims, dst_dims, src_bytes, dst_ptr, kernel_vec, stride_vec, pads_begin,
                    dil_vec,
                );
            }
            other => panic!("MaxPool: unimplemented CPU for DataType {:?}", other),
        }
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }
}
