use crate::ComputeManager;
use crate::error::VKMLError;
use crate::instruction::softmax::push_constants::SoftmaxPushConstants;
use crate::utils::as_bytes;
use crate::{
    gpu::vk_gpu::Gpu,
    instruction::{
        gpu_operations::GPUOperation, instruction::Instruction, softmax::f32_f32_cpu::f32_f32_cpu,
    },
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

#[derive(Clone)]
pub struct SoftmaxInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    pub axis: Option<i64>,
}

impl Debug for SoftmaxInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Softmax(src={}, dst={}, axis={:?})",
            self.src, self.dst, self.axis
        )
    }
}

impl SoftmaxInstruction {
    fn resolve_axis(&self, rank: usize) -> usize {
        let axis = self.axis.unwrap_or(-1);
        if axis < 0 {
            (rank as i64 + axis) as usize
        } else {
            axis as usize
        }
    }
}

impl Instruction for SoftmaxInstruction {
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

        let dims = src_tensor.desc.dims();
        let dim = self.resolve_axis(dims.len());

        // Currently we only support softmax on the last dimension
        assert_eq!(
            dim,
            dims.len() - 1,
            "Only softmax on the last dimension is currently implemented, requested dimension: {}",
            dim
        );

        let feature_size = dims[dim] as usize;
        let batch_size = src_tensor.desc.num_elements() / feature_size;

        // Create push constants struct (compute before GPU ops)
        let push_constants = SoftmaxPushConstants {
            batch_size: batch_size as u32,
            feature_size: feature_size as u32,
        };

        let pc_bytes = as_bytes(&push_constants);

        // Choose operation based on data type
        let src_dtype = src_tensor.desc.data_type();
        let dst_dtype = dst_tensor.desc.data_type();
        let gpu_op = match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => GPUOperation::Softmax_F32_F32,
            (DataType::Float16, DataType::Float16) => GPUOperation::Softmax_F16_F16,
            _ => {
                return Err(VKMLError::Instruction(format!(
                    "GPU Softmax unimplemented for DataType src:{:?}, dst:{:?}",
                    src_dtype, dst_dtype
                )));
            }
        };

        // Choose an optimal local workgroup for per-batch parallelism
        // We want one workgroup per batch; pick local_size sized to GPU limits
        let mut local_size = gpu.optimal_workgroup_size_1d(feature_size as u64);

        // Clamp the X component to shader max local size x (256)
        if local_size[0] > 256 {
            local_size[0] = 256;
        }

        let binding_count = 2; // src, dst

        // Bind pipeline and descriptors (src=0, dst=1)
        gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);
        gpu.bind_storage_buffers(command_buffer, &[src_mem, dst_mem]);

        // Push constants
        gpu.bind_push_constants(command_buffer, binding_count, pc_bytes);

        // Calculate dispatch size based on batch size (one workgroup per batch)
        gpu.dispatch(command_buffer, local_size, [batch_size as u64, 1, 1]);

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        assert!(
            self.src != self.dst,
            "Cannot use Softmax for in-place operation"
        );

        let src_tensor = cm.tensor_read(self.src);
        let dst_tensor = cm.tensor_write(self.dst);

        let dims = src_tensor.desc.dims();
        let dim = self.resolve_axis(dims.len());

        assert_eq!(
            dim,
            dims.len() - 1,
            "CPU Softmax currently only supports the last dimension"
        );

        let src_dtype = src_tensor.desc.data_type();
        let dst_dtype = dst_tensor.desc.data_type();

        let src_bytes = src_tensor.get_cpu_memory_slice_or_panic();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => {
                f32_f32_cpu(dims, dim, src_bytes, dst_ptr);
            }
            _ => unimplemented!(
                "softmax.rs unimplemented cpu instruction for DataType src:{:?}, dst:{:?}",
                src_dtype,
                dst_dtype
            ),
        }
    }
}
