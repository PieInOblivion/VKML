use crate::ComputeManager;
use crate::error::VKMLError;
use crate::instruction::expand::f32_f32_cpu::f32_f32_cpu;
use crate::instruction::expand::push_constants::ExpandPushConstants;
use crate::utils::as_bytes;
use crate::{
    gpu::vk_gpu::Gpu,
    instruction::{gpu_operations::GPUOperation, instruction::Instruction},
    tensor::TensorDesc,
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

pub struct ExpandInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    pub shape_values: Vec<i64>,
}

impl Debug for ExpandInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Expand(src={}, dst={}, shape={:?})",
            self.src, self.dst, self.shape_values
        )
    }
}

impl Instruction for ExpandInstruction {
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
        let src_tensor = cm.tensor_read(self.src);
        let src_mem = src_tensor.get_gpu_memory_or_panic();
        let dst_tensor = cm.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        // Get tensor descriptions
        let src_desc = src_tensor.desc();
        let dst_desc = dst_tensor.desc();

        let src_dims_usize = src_desc.dims();
        let dst_dims_usize = dst_desc.dims();

        let rank = dst_dims_usize.len() as u32;
        assert!(
            rank <= 8,
            "Expand: tensor rank {} exceeds maximum supported rank of 8",
            rank
        );

        let mut dims_arr = [0u32; 8];
        for (i, &d) in dst_dims_usize.iter().enumerate().take(8) {
            dims_arr[i] = d as u32;
        }

        // Calculate broadcast strides for src tensor
        let strides_src_usize = TensorDesc::broadcast_strides(src_dims_usize, dst_dims_usize);

        let mut strides_src_arr = [0u32; 8];
        for (i, &s) in strides_src_usize.iter().enumerate().take(8) {
            strides_src_arr[i] = s as u32;
        }

        let total_elements: u64 = dst_dims_usize.iter().map(|d| *d as u64).product();

        let push_const_values = ExpandPushConstants {
            rank,
            pad: 0,
            total: total_elements as u32,
            dims: dims_arr,
            strides_src: strides_src_arr,
        };

        let push_constant_bytes = as_bytes(&push_const_values);

        // Choose operation based on tensor DataType
        let src_dtype = src_desc.data_type();
        let dst_dtype = dst_desc.data_type();
        let gpu_op = match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => GPUOperation::Expand_F32_F32,
            (DataType::Float16, DataType::Float16) => GPUOperation::Expand_F16_F16,
            _ => {
                return Err(VKMLError::Instruction(format!(
                    "GPU Expand unimplemented for DataType src:{:?}, dst:{:?}",
                    src_dtype, dst_dtype
                )));
            }
        };

        // Optimal local workgroup size for 1D element-wise op
        let local_size = gpu.optimal_workgroup_size_1d(total_elements);

        let binding_count = 2; // src, dst

        // Bind pipeline, storage buffers, push constants
        gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);
        gpu.bind_storage_buffers(command_buffer, &[src_mem, dst_mem]);
        gpu.bind_push_constants(command_buffer, binding_count, push_constant_bytes);

        let num_elements: u64 = dst_dims_usize.iter().map(|d| *d as u64).product();
        gpu.dispatch(command_buffer, local_size, [num_elements, 1, 1]);

        Ok(())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        let src_tensor = cm.tensor_read(self.src);
        let dst_tensor = cm.tensor_write(self.dst);

        let src_dims = src_tensor.desc().dims();
        let dst_dims = dst_tensor.desc().dims().to_vec();

        // Verify that the expand is valid
        // According to ONNX spec, dimensions are right-aligned
        // Two corresponding dimensions must have the same value, or one of them is equal to 1
        let src_rank = src_dims.len();
        let dst_rank = dst_dims.len();

        // Pad src_dims on the left to match dst_rank
        let mut padded_src_dims = vec![1; dst_rank];
        let offset = dst_rank.saturating_sub(src_rank);
        for (i, &dim) in src_dims.iter().enumerate() {
            padded_src_dims[offset + i] = dim;
        }

        // Verify broadcast compatibility
        for i in 0..dst_rank {
            let src_dim = padded_src_dims[i];
            let dst_dim = dst_dims[i];
            if src_dim != dst_dim && src_dim != 1 {
                panic!(
                    "Expand: incompatible shapes src={:?} (padded={:?}), dst={:?}",
                    src_dims, padded_src_dims, dst_dims
                );
            }
        }

        // Calculate broadcast strides
        let strides_src = TensorDesc::broadcast_strides(src_dims, &dst_dims);

        let src_dtype = src_tensor.desc().data_type();
        let dst_dtype = dst_tensor.desc().data_type();

        let src_bytes = src_tensor.get_cpu_memory_slice_or_panic();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => {
                f32_f32_cpu(strides_src, dst_dims, src_bytes, dst_ptr)
            }
            _ => unimplemented!(
                "expand.rs unimplemented cpu instruction for DataType src:{:?}, dst:{:?}",
                src_dtype,
                dst_dtype
            ),
        }
    }
}
