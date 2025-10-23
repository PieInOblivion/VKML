use crate::error::VKMLError;
use crate::{
    ComputeManager,
    gpu::vk_gpu::Gpu,
    instruction::{
        gpu_operations::GPUOperation, instruction::Instruction, relu::f32_f32_cpu::f32_f32_cpu,
    },
    tensor::TensorDesc,
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

#[derive(Clone)]
pub struct ReLUInstruction {
    pub src: TensorId,
    pub dst: TensorId,
}

impl Debug for ReLUInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "ReLU(src={}, dst={})", self.src, self.dst)
    }
}

impl Instruction for ReLUInstruction {
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

        let num_elements = dst_tensor.desc.num_elements() as u64;

        // Choose operation based on DataType
        let src_dtype = src_tensor.desc.data_type();
        let dst_dtype = dst_tensor.desc.data_type();
        let gpu_op = match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => GPUOperation::ReLU_F32_F32,
            (DataType::Float16, DataType::Float16) => GPUOperation::ReLU_F16_F16,
            _ => {
                return Err(VKMLError::Instruction(format!(
                    "GPU ReLU unimplemented for DataType src:{:?}, dst:{:?}",
                    src_dtype, dst_dtype
                )));
            }
        };

        // Choose an optimal local workgroup size for this 1D element-wise op
        let local_size = gpu.optimal_workgroup_size_1d(num_elements);
        let binding_count = 2; // src, dst

        // Bind pipeline first so descriptor push is associated with the correct layout
        gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size, binding_count);

        gpu.bind_storage_buffers(command_buffer, &[src_mem, dst_mem]);

        // Dispatch
        gpu.dispatch(command_buffer, local_size, [num_elements, 1, 1]);

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        // Follow add.rs style: compute broadcast shapes/strides and dispatch to typed helpers
        assert!(
            self.src != self.dst,
            "Cannot use ReLU for in-place operation"
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
                f32_f32_cpu(sa, c, src_bytes, dst_ptr);
            }
            _ => unimplemented!(
                "relu.rs unimplemented cpu instruction for DataType src:{:?}, dst:{:?}",
                src_dtype,
                dst_dtype
            ),
        }
    }
}
