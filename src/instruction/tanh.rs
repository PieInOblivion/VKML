use crate::{
    dataloader::error::VKMLEngineError,
    gpu::{compute_pipelines::GPUMemoryOperation, vk_gpu::GPU},
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use std::{
    fmt::{Debug, Formatter, Result as FmtResult},
    ptr,
};
use vulkanalia::{vk, vk::DeviceV1_0};

use super::instruction::Instruction;

#[derive(Clone)]
pub struct TanhInstruction {
    pub src: TensorId,
    pub dst: TensorId,
}

impl Debug for TanhInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "Tanh(src={}, dst={})", self.src, self.dst)
    }
}

impl Instruction for TanhInstruction {
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
        let src_mem = tensor_graph.get_gpu_memory_or_panic(&self.src);
        let dst_mem = tensor_graph.get_gpu_memory_or_panic(&self.dst);

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                inheritance_info: ptr::null(),
            };

            gpu.get_device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            let set_layouts = [*gpu.get_descriptor_set_layout()];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                next: ptr::null(),
                descriptor_pool: *gpu.get_descriptor_pool(),
                descriptor_set_count: 1,
                set_layouts: set_layouts.as_ptr(),
            };

            let descriptor_set = gpu.get_device().allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_infos = [
                // src buffer (binding 0)
                vk::DescriptorBufferInfo {
                    buffer: src_mem.buffer,
                    offset: 0,
                    range: src_mem.size,
                },
                // dst buffer (binding 2)
                vk::DescriptorBufferInfo {
                    buffer: dst_mem.buffer,
                    offset: 0,
                    range: dst_mem.size,
                },
            ];

            let write_descriptor_sets = [
                // src buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[0],
                    image_info: ptr::null(),
                    texel_buffer_view: ptr::null(),
                },
                // dst buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[1],
                    image_info: ptr::null(),
                    texel_buffer_view: ptr::null(),
                },
            ];

            gpu.get_device()
                .update_descriptor_sets(&write_descriptor_sets, &[] as &[vk::CopyDescriptorSet]);

            let pipeline = gpu
                .get_compute_pipelines()
                .get_pipeline(GPUMemoryOperation::Tanh)
                .ok_or(format!("{:?} pipeline not found", GPUMemoryOperation::Tanh))?;

            gpu.get_device().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );

            gpu.get_device().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                gpu.get_compute_pipelines().get_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            let workgroup_size = 256;
            let num_elements = dst_mem.size / std::mem::size_of::<f32>() as u64;
            let num_workgroups = (num_elements + workgroup_size as u64 - 1) / workgroup_size as u64;

            gpu.get_device()
                .cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

            gpu.get_device().end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, tensor_graph: &mut TensorGraph) -> Result<(), VKMLEngineError> {
        let src_data = tensor_graph.tensors[self.src].data.borrow_cpu_data()?;
        let mut dst_data = tensor_graph.tensors[self.dst].data.borrow_mut_cpu_data()?;

        // Verify tensor sizes
        if dst_data.len() != src_data.len() {
            return Err(VKMLEngineError::ShapeMismatch(format!(
                "Destination tensor size {} doesn't match source tensor size {}",
                dst_data.len(),
                src_data.len()
            )));
        }

        // Update in-place
        for i in 0..src_data.len() {
            dst_data[i] = src_data[i].tanh();
        }

        Ok(())
    }
}
