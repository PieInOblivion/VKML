use crate::ComputeManager;
use crate::instruction::init_he::push_constants::InitHePushConstants;
use crate::utils::as_bytes;
use crate::{
    gpu::vk_gpu::Gpu,
    instruction::{
        gpu_operations::GPUMemoryOperation, init_he::f32_cpu::f32_cpu, instruction::Instruction,
    },
    tensor_graph::tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk::Handle;
use vulkanalia::vk::KhrPushDescriptorExtension;
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
        gpu: &Gpu,
        command_buffer: vk::CommandBuffer,
        cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dst_tensor = cm.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            gpu.get_device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            let buffer_info = vk::DescriptorBufferInfo {
                buffer: dst_mem.buffer,
                offset: 0,
                range: dst_mem.size,
            };
            let write_descriptor_set = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                next: std::ptr::null(),
                dst_set: vk::DescriptorSet::null(),
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                buffer_info: &buffer_info,
                image_info: std::ptr::null(),
                texel_buffer_view: std::ptr::null(),
            };

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
            gpu.get_device().cmd_push_descriptor_set_khr(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                gpu.get_layout(),
                0,
                &[write_descriptor_set],
            );

            let dst_elems = dst_mem.size / std::mem::size_of::<f32>() as u64;
            let (fan_in, _) = dst_tensor.desc.calculate_fan_in_out();
            let seed = rand::random::<u32>();
            let gain = 2.0f32; // default gain for He init (variance scaling)

            let push_constants = InitHePushConstants {
                total_elements: dst_elems as u32,
                fan_in: fan_in as u32,
                seed,
                gain,
            };

            let pc_bytes = as_bytes(&push_constants);

            gpu.get_device().cmd_push_constants(
                command_buffer,
                gpu.get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            let workgroup_size = 256u32;
            let num_workgroups = (dst_elems as u32).div_ceil(workgroup_size);
            gpu.get_device()
                .cmd_dispatch(command_buffer, num_workgroups, 1, 1);

            gpu.get_device().end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        let dst = cm.tensor_write(self.dst);
        let (fan_in, _) = dst.desc.calculate_fan_in_out();
        // compute immutable info before taking mutable borrow
        let dst_dims = dst.desc.dims().to_vec();
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
