use crate::ComputeManager;
use crate::instruction::softmax::push_constants::SoftmaxPushConstants;
use crate::utils::as_bytes;
use crate::{
    gpu::vk_gpu::GPU,
    instruction::{
        gpu_operations::GPUMemoryOperation, instruction::Instruction, softmax::f32_cpu::f32_cpu,
    },
    tensor_graph::tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::{
    fmt::{Debug, Formatter, Result as FmtResult},
    ptr,
};
use vulkanalia::{vk, vk::DeviceV1_0};

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
        cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let src_tensor = cm.tensor_read(self.src);
        let src_mem = src_tensor.get_gpu_memory_or_panic();
        let dst_tensor = cm.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        // Currently we only support softmax on the last dimension
        assert_eq!(
            self.dim,
            src_tensor.desc.to_dims().len() - 1,
            "Only softmax on the last dimension is currently implemented, requested dimension: {}",
            self.dim
        );

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

            let op_datatype = src_tensor.desc.data_type();
            let gpu_op = match op_datatype {
                DataType::Float => GPUMemoryOperation::Softmax_F32,
                _ => {
                    return Err(format!(
                        "GPU Softmax unimplemented for DataType {:?}",
                        op_datatype
                    )
                    .into());
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

            let feature_size = src_tensor.desc.to_dims()[self.dim] as usize;
            let batch_size = src_mem.size as usize / std::mem::size_of::<f32>() / feature_size;

            // Create push constants struct (shared definition)
            let push_constants = SoftmaxPushConstants {
                batch_size: batch_size as u32,
                feature_size: feature_size as u32,
            };

            let pc_bytes = as_bytes(&push_constants);

            gpu.get_device().cmd_push_constants(
                command_buffer,
                gpu.get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            // Calculate dispatch size based on batch size
            // One workgroup per batch for now
            let num_workgroups = (batch_size as u64 + 255) / 256;

            gpu.get_device()
                .cmd_dispatch(command_buffer, num_workgroups as u32, 1, 1);

            gpu.get_device().end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        assert!(
            self.src != self.dst,
            "Cannot use Softmax for in-place operation"
        );

        let src_tensor = cm.tensor_read(self.src);
        let mut dst_tensor = cm.tensor_write(self.dst);

        let dims = src_tensor.desc.to_dims();
        assert_eq!(
            self.dim,
            dims.len() - 1,
            "CPU Softmax currently only supports the last dimension"
        );

        let src_bytes = src_tensor.get_cpu_memory_slice_or_panic();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match src_tensor.desc.data_type() {
            DataType::Float => {
                f32_cpu(dims, self.dim, src_bytes, dst_ptr);
            }
            _ => unimplemented!(
                "softmax.rs unimplemented cpu instruction for DataType {:?}",
                src_tensor.desc.data_type()
            ),
        }
    }
}
