use crate::error::VKMLError;
use crate::gpu::vk_gpu::Gpu;
use crate::instruction::reducemean::f32_cpu::f32_cpu;
use crate::instruction::reducemean::push_constants::ReduceMeanPushConstants;
use crate::instruction::{GPUOperation, Instruction};
use crate::utils::as_bytes;
use crate::{ComputeManager, tensor::TensorDesc, tensor_graph::TensorId};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

pub struct ReduceMeanInstruction {
    pub src: TensorId,
    pub axes: Option<Vec<i64>>,
    pub keepdims: i64,
    pub noop_with_empty_axes: i64,
    pub dst: TensorId,
}

impl Debug for ReduceMeanInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "ReduceMean(src={}, axes={:?}, keepdims={}, dst={})",
            self.src, self.axes, self.keepdims, self.dst
        )
    }
}

impl Instruction for ReduceMeanInstruction {
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
        // GPU implementation: two-pass reduction (sum then scale)
        let src_t = cm.tensor_read(self.src);
        let src_mem = src_t.get_gpu_memory_or_panic();
        let dst_t = cm.tensor_read(self.dst);
        let dst_mem = dst_t.get_gpu_memory_or_panic();

        // Determine axes to reduce
        let rank = src_t.desc.ndim() as i64;
        let axes_vec: Vec<i64> = if let Some(a) = &self.axes {
            a.clone()
        } else if self.noop_with_empty_axes != 0 {
            Vec::new()
        } else {
            (0..rank).collect()
        };

        // If noop and empty axes, nothing to do on GPU (copy handled elsewhere)
        if axes_vec.is_empty() && self.noop_with_empty_axes != 0 {
            return Ok(());
        }

        // compute total elements and reduction_size and output elements
        let _total_elements: u64 = src_t.desc.dims().iter().map(|d| *d as u64).product();
        let mut reduction_size: u64 = 1;
        for &a in &axes_vec {
            reduction_size *= src_t.desc.dims()[a as usize] as u64;
        }

        // compute output dims and elements
        let mut out_dims: Vec<i64> = Vec::new();
        for (i, &d) in src_t.desc.dims().iter().enumerate() {
            if axes_vec.contains(&(i as i64)) {
                if self.keepdims != 0 {
                    out_dims.push(1);
                }
            } else {
                out_dims.push(d);
            }
        }
        if out_dims.is_empty() {
            out_dims.push(1);
        }
        let out_elements: u64 = out_dims.iter().map(|d| *d as u64).product();

        let mean_pc = ReduceMeanPushConstants {
            total: out_elements as u32,
            reduction_size: reduction_size as u32,
        };
        let mean_pc_bytes = as_bytes(&mean_pc);

        let src_dtype = src_t.desc.data_type();
        let dst_dtype = dst_t.desc.data_type();

        // Select GPUOperation based on DataType trio (src,dst)
        let gpu_op = match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => GPUOperation::ReduceMean_F32,
            _ => {
                return Err(VKMLError::Instruction(format!(
                    "GPU ReduceMean unimplemented for DataType src:{:?}, dst:{:?}",
                    src_dtype, dst_dtype
                )));
            }
        };

        // Choose a local size for dispatch (1D op)
        let local_size = gpu.optimal_workgroup_size_1d(out_elements);
        let binding_count = 2; // src, dst

        gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);
        gpu.bind_storage_buffers(command_buffer, &[src_mem, dst_mem]);
        gpu.bind_push_constants(command_buffer, binding_count, mean_pc_bytes);
        gpu.dispatch(command_buffer, local_size, [out_elements, 1, 1]);

        Ok(())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        // Basic CPU implementation: compute mean over axes
        let src_t = cm.tensor_read(self.src);
        let src_desc = src_t.desc.clone();
        let src_dims = src_desc.dims().to_vec();
        let rank = src_dims.len() as i64;

        // Determine axes to reduce
        let axes_vec: Vec<i64> = if let Some(a) = &self.axes {
            a.clone()
        } else {
            // No axes => reduce over all axes unless noop_with_empty_axes == 1
            if self.noop_with_empty_axes != 0 {
                Vec::new()
            } else {
                (0..rank).collect()
            }
        };

        // If noop_with_empty_axes==1 and axes empty => copy input to output
        if axes_vec.is_empty() && self.noop_with_empty_axes != 0 {
            let src_bytes = src_t.get_cpu_memory_slice_or_panic();
            let dst_t = cm.tensor_write(self.dst);
            dst_t.desc = src_desc.clone();
            dst_t
                .get_cpu_memory_mut_slice_or_panic()
                .copy_from_slice(src_bytes);
            return;
        }

        // Compute output shape
        let keep = self.keepdims != 0;
        let mut out_dims: Vec<i64> = Vec::new();
        for (i, &d) in src_dims.iter().enumerate() {
            if axes_vec.contains(&(i as i64)) {
                if keep {
                    out_dims.push(1);
                }
            } else {
                out_dims.push(d);
            }
        }
        if !keep && out_dims.is_empty() {
            out_dims.push(1); // scalar -> 1-element tensor representation
        }

        // Update dst descriptor
        let dst_t = cm.tensor_write(self.dst);
        dst_t.desc = TensorDesc::new(out_dims.clone(), src_desc.data_type());

        let src_bytes = src_t.get_cpu_memory_slice_or_panic();
        let out_bytes = dst_t.get_cpu_memory_mut_slice_or_panic();

        // For simplicity, only implement numeric float32 path
        match src_desc.data_type() {
            DataType::Float => {
                f32_cpu(src_bytes, &src_dims, &axes_vec, keep, out_bytes);
            }
            _ => unimplemented!(
                "ReduceMean CPU: DataType {:?} not implemented",
                src_desc.data_type()
            ),
        }
    }
}
