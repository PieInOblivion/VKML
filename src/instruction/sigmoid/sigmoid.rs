use crate::error::VKMLError;
use crate::{
    ComputeManager,
    gpu::vk_gpu::Gpu,
    instruction::{
        gpu_operations::GPUOperation, instruction::Instruction, sigmoid::f32_f32_cpu::f32_f32_cpu,
    },
    tensor::TensorDesc,
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

pub struct SigmoidInstruction {
    pub src: TensorId,
    pub dst: TensorId,
}

impl Debug for SigmoidInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "Sigmoid(src={}, dst={})", self.src, self.dst)
    }
}

impl Instruction for SigmoidInstruction {
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

        // Prepare CPU-side values
        let num_elements = dst_mem.size / std::mem::size_of::<f32>() as u64;

        // Choose operation based on data type
        let src_dtype = src_tensor.desc.data_type();
        let dst_dtype = dst_tensor.desc.data_type();
        let gpu_op = match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => GPUOperation::Sigmoid_F32_F32,
            _ => {
                return Err(VKMLError::Instruction(format!(
                    "GPU Sigmoid unimplemented for DataType src:{:?}, dst:{:?}",
                    src_dtype, dst_dtype
                )));
            }
        };

        // Choose local workgroup and bind pipeline
        let local_size = gpu.optimal_workgroup_size_1d(num_elements);
        let binding_count = 2; // src, dst

        gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);
        gpu.bind_storage_buffers(command_buffer, &[src_mem, dst_mem]);

        gpu.dispatch(command_buffer, local_size, [num_elements, 1, 1]);

        Ok(())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        assert!(
            self.src != self.dst,
            "Cannot use Sigmoid for in-place operation"
        );

        let src_tensor = cm.tensor_read(self.src);
        let dst_tensor = cm.tensor_write(self.dst);

        let a = src_tensor.desc.dims();
        let c = dst_tensor.desc.dims().to_vec();

        let bc = TensorDesc::broadcast_shape(a, &c)
            .unwrap_or_else(|| panic!("Can't broadcast {:?} vs {:?}", a, c));
        assert_eq!(bc.as_slice(), c, "Broadcast {:?} != dst {:?}", bc, c);

        let sa = TensorDesc::broadcast_strides(a, &c);

        let src_dtype = src_tensor.desc.data_type();
        let dst_dtype = dst_tensor.desc.data_type();

        let src_bytes = src_tensor.get_cpu_memory_slice_or_panic();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => {
                f32_f32_cpu(sa, vec![], c, src_bytes, &[], dst_ptr)
            }
            _ => unimplemented!(
                "sigmoid.rs unimplemented cpu instruction for DataType src:{:?}, dst:{:?}",
                src_dtype,
                dst_dtype
            ),
        }
    }
}
