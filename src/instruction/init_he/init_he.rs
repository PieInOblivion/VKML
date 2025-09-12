use crate::{
    gpu::vk_gpu::GPU,
    instruction::{
        gpu_operations::GPUMemoryOperation, init_he::f32_cpu::f32_cpu, instruction::Instruction,
    },
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::{vk, vk::DeviceV1_0};

#[derive(Clone)]
pub struct InitHeInstruction {
    pub dst: TensorId,
}

impl Debug for InitHeInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "InitHe(dst={})", self.dst)
    }
}

impl Instruction for InitHeInstruction {
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

            let op_datatype = dst_tensor.desc.data_type();
            let gpu_op = match op_datatype {
                onnx_extractor::DataType::Float => GPUMemoryOperation::InitHe_F32,
                _ => {
                    return Err(
                        format!("GPU InitHe unimplemented for DataType {:?}", op_datatype).into(),
                    );
                }
            };

            let pipeline = gpu.get_or_create_pipeline(gpu_op);
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
            let (fan_in, _) = dst_tensor.desc.calculate_fan_in_out();
            let seed = rand::random::<u32>();
            let gain = 2.0f32; // default gain for He init (variance scaling)

            #[repr(C)]
            struct PC {
                total_elements: u32,
                fan_in: u32,
                seed: u32,
                gain: f32,
            }

            let push_constants = PC {
                total_elements: dst_elems as u32,
                fan_in: fan_in as u32,
                seed,
                gain,
            };

            gpu.get_device().cmd_push_constants(
                command_buffer,
                gpu.get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const PC as *const u8,
                    std::mem::size_of::<PC>(),
                ),
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
        let (fan_in, _) = dst.desc.calculate_fan_in_out();
        // compute immutable info before taking mutable borrow
        let dst_dims = dst.desc.to_dims();
        let dtype = dst.desc.data_type();
        let out = dst.get_cpu_memory_mut_slice_or_panic();

        match dtype {
            DataType::Float => {
                f32_cpu(fan_in, dst_dims, out);
            }
            _ => unimplemented!("InitHe CPU for other types"),
        }
    }
}
