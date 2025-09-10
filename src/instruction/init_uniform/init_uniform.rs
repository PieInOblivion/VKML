use crate::{
    gpu::vk_gpu::GPU,
    instruction::instruction::Instruction,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use rand::distr::{Distribution, Uniform};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::{vk, vk::DeviceV1_0};

#[derive(Clone)]
pub struct InitUniformInstruction {
    pub dst: TensorId,
    pub min: f32,
    pub max: f32,
}

impl Debug for InitUniformInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "InitUniform(dst={}, min={}, max={})",
            self.dst, self.min, self.max
        )
    }
}

impl Instruction for InitUniformInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        vec![]
    }
    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }
    fn remap_tensor_ids(&mut self, _new_inputs: &[TensorId], new_outputs: &[TensorId]) {
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
        let dst_tensor = tensor_graph.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                ..Default::default()
            };
            gpu.get_device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            let set_layouts = [*gpu.get_descriptor_set_layout()];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptor_pool: *gpu.get_descriptor_pool(),
                descriptor_set_count: 1,
                set_layouts: set_layouts.as_ptr(),
                ..Default::default()
            };
            let descriptor_set = gpu.get_device().allocate_descriptor_sets(&alloc_info)?[0];

            let buffer_info = vk::DescriptorBufferInfo {
                buffer: dst_mem.buffer,
                offset: 0,
                range: dst_mem.size,
            };
            let write_descriptor_set = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                buffer_info: &buffer_info,
                ..Default::default()
            };
            gpu.get_device()
                .update_descriptor_sets(&[write_descriptor_set], &[] as &[vk::CopyDescriptorSet]);

            let pipeline = gpu.get_or_create_pipeline(
                crate::instruction::gpu_operations::GPUMemoryOperation::InitUniform_F32,
            );
            gpu.get_device().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            gpu.get_device().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                gpu.get_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            let dst_elems = dst_mem.size / std::mem::size_of::<f32>() as u64;
            let mut pc = [0f32; 4];
            pc[0] = dst_elems as f32;
            pc[1] = self.min;
            pc[2] = self.max;
            pc[3] = rand::random::<u32>() as f32;
            let pc_bytes = std::slice::from_raw_parts(
                pc.as_ptr() as *const u8,
                std::mem::size_of::<[f32; 4]>(),
            );
            gpu.get_device().cmd_push_constants(
                command_buffer,
                gpu.get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            let workgroup_size = 256u32;
            let num_workgroups = ((dst_elems as u32) + workgroup_size - 1) / workgroup_size;
            gpu.get_device()
                .cmd_dispatch(command_buffer, num_workgroups, 1, 1);

            gpu.get_device().end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, tensor_graph: &TensorGraph) {
        let mut dst = tensor_graph.tensor_write(self.dst);
        let total = dst.desc.num_elements();

        match dst.desc.data_type() {
            onnx_extractor::DataType::Float => {
                let out = dst.get_cpu_memory_mut_slice_or_panic();
                let dist = Uniform::new(self.min, self.max);
                let mut rng = rand::rng();
                for i in 0..total {
                    let v: f32 = dist.unwrap().sample(&mut rng);
                    let bytes = v.to_le_bytes();
                    let base = i * 4;
                    out[base..base + 4].copy_from_slice(&bytes);
                }
            }
            _ => unimplemented!("InitUniform CPU for other types"),
        }
    }
}
