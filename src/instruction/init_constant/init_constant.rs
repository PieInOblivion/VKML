use crate::ComputeManager;
use crate::instruction::init_constant::push_constants::InitConstantPushConstants;
use crate::utils::as_bytes;
use crate::{
    gpu::vk_gpu::GPU,
    instruction::{gpu_operations::GPUMemoryOperation, instruction::Instruction},
    tensor_graph::tensor_graph::TensorId,
};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::{vk, vk::DeviceV1_0};

#[derive(Clone)]
pub struct InitConstantInstruction {
    pub dst: TensorId,
    pub constant: Vec<u8>, // raw bytes, little endian
}

impl Debug for InitConstantInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "InitConstant(dst={}, constant_len={})",
            self.dst,
            self.constant.len()
        )
    }
}

impl Instruction for InitConstantInstruction {
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
        cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dst_tensor = cm.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        // Use generic constant-init compute shader that writes elements up to 8 bytes
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

            // Use the generic InitConstant GPU operation for supported sizes (1..=8 bytes)
            let pipeline = gpu.get_or_create_pipeline(GPUMemoryOperation::InitConstant);

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

            // Determine element size for this tensor and pass it to the GPU so the shader
            // can write using the correct stride/width. Panic if we don't know the size.
            let elem_size_usize = dst_tensor.desc.data_type().size_in_bytes().expect(&format!(
                "InitConstant create_command_buffer: unknown element size for DataType {:?}",
                dst_tensor.desc.data_type()
            ));
            let elem_size = elem_size_usize as u32;

            // Validate element size (host-side defense). We support up to 8 bytes per element.
            if elem_size_usize == 0 || elem_size_usize > 8 {
                panic!(
                    "InitConstant create_command_buffer: unsupported element size {} (must be 1..=8)",
                    elem_size_usize
                );
            }

            // Compute total bytes from the tensor descriptor (preferred source of truth)
            let total_bytes_usize = dst_tensor.desc.size_in_bytes();
            if total_bytes_usize == 0 {
                // nothing to do
                return Ok(());
            }
            if total_bytes_usize > u32::MAX as usize {
                panic!(
                    "InitConstant create_command_buffer: total bytes {} too large (must fit in u32)",
                    total_bytes_usize
                );
            }

            let total_words = ((total_bytes_usize + 3) / 4) as u32;

            // Build a little-endian u64 from the first up to 8 bytes of `self.constant`.
            // Use `.get()` with a default of 0 so GPU path is tolerant of shorter inputs.
            let mut value_u64: u64 = 0;
            for i in 0..8usize {
                let b = *self.constant.get(i).unwrap_or(&0) as u64;
                value_u64 |= b << (8 * i);
            }

            let push_constants = InitConstantPushConstants {
                elem_size,
                value_lo: (value_u64 & 0xFFFFFFFF) as u32,
                value_hi: (value_u64 >> 32) as u32,
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
            let num_workgroups = ((total_words as u32) + workgroup_size - 1) / workgroup_size;
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
        let mut dst = cm.tensor_write(self.dst);
        let dtype = dst.desc.data_type();

        // Use DataType's helper to get element size in bytes; panic if unknown so we don't assume a size.
        let required_elem_bytes: usize = dtype.size_in_bytes().expect(&format!(
            "InitConstant execute_cpu: unknown element size for DataType {:?}",
            dtype
        ));
        assert!(required_elem_bytes > 0);

        // Require that the provided constant bytes contain at least one element.
        if self.constant.len() < required_elem_bytes {
            panic!(
                "InitConstant execute_cpu: constant bytes length {} too small for DataType {:?} (need {} bytes)",
                self.constant.len(),
                dtype,
                required_elem_bytes
            );
        }

        // Use the first `required_elem_bytes` bytes as the element pattern (little-endian assumed).
        let pattern = &self.constant[0..required_elem_bytes];

        let out = dst.get_cpu_memory_mut_slice_or_panic();

        // Destination length must be a multiple of the element size.
        if out.len() % required_elem_bytes != 0 {
            panic!(
                "InitConstant execute_cpu: destination length {} is not a multiple of element size {}",
                out.len(),
                required_elem_bytes
            );
        }

        for chunk in out.chunks_mut(required_elem_bytes) {
            chunk.copy_from_slice(pattern);
        }
    }
}
