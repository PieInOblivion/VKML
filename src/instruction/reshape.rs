use crate::{
    gpu::vk_gpu::GPU,
    tensor::tensor_desc::TensorDesc,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::{vk, vk::DeviceV1_0};

use super::instruction::Instruction;

#[derive(Clone)]
pub struct ReshapeInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    pub new_shape: TensorDesc,
}

impl Debug for ReshapeInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Reshape(src={}, dst={}, shape={:?})",
            self.src,
            self.dst,
            self.new_shape.to_dims()
        )
    }
}

impl Instruction for ReshapeInstruction {
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
        gpu: &GPU,
        command_buffer: vk::CommandBuffer,
        tensor_graph: &TensorGraph,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Reshape in Vulkan is a logical operation, not a physical one
        // We essentially need to copy data between the tensors
        let src_mem = tensor_graph.get_gpu_memory_or_panic(self.src);
        let dst_mem = tensor_graph.get_gpu_memory_or_panic(self.dst);

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                next: std::ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                inheritance_info: std::ptr::null(),
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

    fn execute_cpu(&self, tensor_graph: &mut TensorGraph) {
        let src_data = tensor_graph.tensors[self.src].data.read_data();
        let mut dst_data = tensor_graph.tensors[self.dst].data.write_data();

        assert_eq!(
            dst_data.len(),
            src_data.len(),
            "Reshape: destination tensor size {} doesn't match source tensor size {}",
            dst_data.len(),
            src_data.len()
        );

        dst_data.copy_from_slice(&src_data);
    }
}
