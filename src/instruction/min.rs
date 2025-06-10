use crate::{
    dataloader::error::VKMLEngineError,
    gpu::{compute_pipelines::GPUMemoryOperation, vk_gpu::GPU},
    tensor::tensor_desc::TensorDesc,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::{vk, vk::DeviceV1_0};

use super::instruction::Instruction;

#[derive(Clone)]
pub struct MinInstruction {
    pub src1: TensorId,
    pub src2: TensorId,
    pub dst: TensorId,
}

impl Debug for MinInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Min(src1={}, src2={}, dst={})",
            self.src1, self.src2, self.dst
        )
    }
}

impl Instruction for MinInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.src1, self.src2]
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if new_inputs.len() >= 2 {
            self.src1 = new_inputs[0];
            self.src2 = new_inputs[1];
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
        let src1_mem = tensor_graph.get_gpu_memory_or_panic(&self.src1);
        let src2_mem = tensor_graph.get_gpu_memory_or_panic(&self.src2);
        let dst_mem = tensor_graph.get_gpu_memory_or_panic(&self.dst);

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                next: std::ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                inheritance_info: std::ptr::null(),
            };

            gpu.get_device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            let set_layouts = [*gpu.get_descriptor_set_layout()];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                next: std::ptr::null(),
                descriptor_pool: *gpu.get_descriptor_pool(),
                descriptor_set_count: 1,
                set_layouts: set_layouts.as_ptr(),
            };

            let descriptor_set = gpu.get_device().allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_infos = [
                // src1 buffer (binding 0)
                vk::DescriptorBufferInfo {
                    buffer: src1_mem.buffer,
                    offset: 0,
                    range: src1_mem.size,
                },
                // src2 buffer (binding 1)
                vk::DescriptorBufferInfo {
                    buffer: src2_mem.buffer,
                    offset: 0,
                    range: src2_mem.size,
                },
                // dst buffer (binding 2)
                vk::DescriptorBufferInfo {
                    buffer: dst_mem.buffer,
                    offset: 0,
                    range: dst_mem.size,
                },
            ];

            let write_descriptor_sets = [
                // src1 buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[0],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
                // src2 buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[1],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
                // dst buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[2],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
            ];

            gpu.get_device()
                .update_descriptor_sets(&write_descriptor_sets, &[] as &[vk::CopyDescriptorSet]);

            let pipeline = gpu
                .get_compute_pipelines()
                .get_pipeline(GPUMemoryOperation::Minimum)
                .ok_or(format!(
                    "{:?} pipeline not found",
                    GPUMemoryOperation::Minimum
                ))?;

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
        // First check that this isn't being used as an in-place operation
        if self.src1 == self.dst || self.src2 == self.dst {
            return Err(VKMLEngineError::VulkanLoadError(
                "Cannot use Min for in-place operation. Use MinInplace instead.".to_string(),
            ));
        }

        let src1 = &tensor_graph.tensors[self.src1];
        let src2 = &tensor_graph.tensors[self.src2];
        let dst = &tensor_graph.tensors[self.dst];

        let a = src1.desc.to_dims();
        let b = src2.desc.to_dims();
        let c = dst.desc.to_dims();

        // 1) compute broadcast shape
        let bc = TensorDesc::broadcast_shape(&a, &b).ok_or_else(|| {
            VKMLEngineError::ShapeMismatch(format!("Can't broadcast {:?} vs {:?}", a, b))
        })?;
        // 2) must match dst
        if bc != c {
            return Err(VKMLEngineError::ShapeMismatch(format!(
                "Broadcast {:?} != dst {:?}",
                bc, c
            )));
        }

        let sa = TensorDesc::broadcast_strides(&a, &c);
        let sb = TensorDesc::broadcast_strides(&b, &c);

        let mut dd = dst.data.borrow_mut_cpu_data()?;
        let d1 = src1.data.borrow_cpu_data()?;
        let d2 = src2.data.borrow_cpu_data()?;

        // Simplified loop without aliasing checks
        for i in 0..dd.len() {
            let idxs = TensorDesc::unravel(i, &c);
            let off1 = TensorDesc::offset(&idxs, &sa);
            let off2 = TensorDesc::offset(&idxs, &sb);
            dd[i] = d1[off1].min(d2[off2]);
        }
        Ok(())
    }
}
