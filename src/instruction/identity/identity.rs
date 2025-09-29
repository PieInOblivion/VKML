use crate::{
    ComputeManager, gpu::vk_gpu::Gpu, instruction::instruction::Instruction,
    tensor_graph::tensor_graph::TensorId,
};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::{vk, vk::DeviceV1_0};

#[derive(Clone)]
pub struct IdentityInstruction {
    pub src: TensorId,
    pub dst: TensorId,
}

impl Debug for IdentityInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "Copy(src={}, dst={})", self.src, self.dst)
    }
}

impl Instruction for IdentityInstruction {
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
        let src_tensor = cm.tensor_read(self.src);
        let src_mem = src_tensor.get_gpu_memory_or_panic();
        let dst_tensor = cm.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };

            gpu.get_device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            // Copy regions - entire buffer
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: src_mem.size,
            };

            gpu.get_device().cmd_copy_buffer(
                command_buffer,
                src_mem.buffer,
                dst_mem.buffer,
                &[copy_region],
            );

            gpu.get_device().end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        let src_tensor = cm.tensor_read(self.src);
        let dst_tensor = cm.tensor_write(self.dst);

        dst_tensor.write(&src_tensor.read());
    }
}
