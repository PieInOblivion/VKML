use crate::{
    dataloader::error::VKMLEngineError,
    gpu::{compute_pipelines::GPUMemoryOperation, vk_gpu::GPU},
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use ash::vk;
use std::{
    fmt::{Debug, Formatter, Result as FmtResult},
    ptr,
};

use super::instruction::Instruction;

#[derive(Clone)]
pub struct SoftmaxInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    pub dim: usize,
}

impl Debug for SoftmaxInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Softmax(src={}, dst={}, dim={})",
            self.src, self.dst, self.dim
        )
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

    fn create_command_buffer(
        &self,
        gpu: &GPU,
        command_buffer: vk::CommandBuffer,
        tensor_graph: &TensorGraph,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let src_mem = tensor_graph.get_gpu_memory_or_panic(&self.src);
        let dst_mem = tensor_graph.get_gpu_memory_or_panic(&self.dst);
        let tensor = tensor_graph.tensors.get(self.src).unwrap();

        // Currently we only support softmax on the last dimension
        if self.dim != tensor.desc.to_dims().len() - 1 {
            return Err(format!("Only softmax on the last dimension is currently implemented, requested dimension: {}", self.dim).into());
        }

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: ptr::null(),
                _marker: std::marker::PhantomData,
            };

            gpu.get_device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            let set_layouts = [*gpu.get_descriptor_set_layout()];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: ptr::null(),
                descriptor_pool: *gpu.get_descriptor_pool(),
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                _marker: std::marker::PhantomData,
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
                    p_next: ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[0],
                    p_image_info: ptr::null(),
                    p_texel_buffer_view: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                // dst buffer descriptor
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[1],
                    p_image_info: ptr::null(),
                    p_texel_buffer_view: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            ];

            gpu.get_device()
                .update_descriptor_sets(&write_descriptor_sets, &[]);

            let pipeline = gpu
                .get_compute_pipelines()
                .get_pipeline(GPUMemoryOperation::Softmax)
                .ok_or("Softmax pipeline not found")?;

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

            let feature_size = tensor.desc.to_dims()[self.dim];
            let batch_size = src_mem.size as usize / std::mem::size_of::<f32>() / feature_size;

            // Create push constants struct
            #[repr(C)]
            struct SoftmaxPushConstants {
                batch_size: u32,
                feature_size: u32,
            }

            let push_constants = SoftmaxPushConstants {
                batch_size: batch_size as u32,
                feature_size: feature_size as u32,
            };

            // Push constants to the shader
            gpu.get_device().cmd_push_constants(
                command_buffer,
                gpu.get_compute_pipelines().get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const SoftmaxPushConstants as *const u8,
                    std::mem::size_of::<SoftmaxPushConstants>(),
                ),
            );

            // Calculate dispatch size based on batch size
            // One workgroup per batch for now
            let num_workgroups = (batch_size as u64 + 255) / 256;

            gpu.get_device()
                .cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

            gpu.get_device().end_command_buffer(command_buffer)?;

            Ok(())
        }
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

        let tensor = &tensor_graph.tensors[self.src];
        let dims = tensor.desc.to_dims();

        // CPU implementation currently only supports softmax on the last dimension
        if self.dim != dims.len() - 1 {
            return Err(VKMLEngineError::VulkanLoadError(
                "CPU Softmax currently only supports the last dimension".to_string(),
            ));
        }

        let feature_size = dims[self.dim];
        let batch_size = src_data.len() / feature_size;

        // Process each batch separately
        for b in 0..batch_size {
            let offset = b * feature_size;

            // Find max for numerical stability
            let mut max_val = f32::MIN;
            for i in 0..feature_size {
                max_val = max_val.max(src_data[offset + i]);
            }

            // Compute exponentials and sum
            let mut sum = 0.0;
            for i in 0..feature_size {
                let exp_val = (src_data[offset + i] - max_val).exp();
                dst_data[offset + i] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for i in 0..feature_size {
                dst_data[offset + i] /= sum;
            }
        }

        Ok(())
    }
}
