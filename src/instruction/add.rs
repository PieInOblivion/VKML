use crate::{
    gpu::{compute_pipelines::GPUMemoryOperation, vk_gpu::GPU},
    tensor::tensor_desc::TensorDesc,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::{vk, vk::DeviceV1_0};

use super::instruction::Instruction;

#[derive(Clone)]
pub struct AddInstruction {
    pub src1: TensorId,
    pub src2: TensorId,
    pub dst: TensorId,
}

impl Debug for AddInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Add(src1={}, src2={}, dst={})",
            self.src1, self.src2, self.dst
        )
    }
}

impl Instruction for AddInstruction {
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

        // Get tensor descriptions for calculating broadcast shapes and strides
        let src1_desc = &tensor_graph.tensors[self.src1].desc;
        let src2_desc = &tensor_graph.tensors[self.src2].desc;
        let dst_desc = &tensor_graph.tensors[self.dst].desc;

        let src1_dims_usize = src1_desc.to_dims();
        let src2_dims_usize = src2_desc.to_dims();
        let dst_dims_usize = dst_desc.to_dims();

        // Prepare push constant data
        #[repr(C)]
        struct PushConstants {
            rank: u32,
            pad: u32, // Matches "uint pad;" in shader
            dims: [u32; 8],
            strides_a: [u32; 8],
            strides_b: [u32; 8],
        }

        let rank = dst_dims_usize.len() as u32;
        assert!(
            rank <= 8,
            "Add: tensor rank {} exceeds maximum supported rank of 8",
            rank
        );

        let mut dims_arr = [0u32; 8];
        for i in 0..dst_dims_usize.len() {
            dims_arr[i] = dst_dims_usize[i] as u32;
        }

        // Calculate broadcast shape and strides (similar to execute_cpu)
        let broadcast_dims = TensorDesc::broadcast_shape(&src1_dims_usize, &src2_dims_usize)
            .expect(&format!(
                "GPU Add: Can't broadcast {:?} vs {:?}",
                src1_dims_usize, src2_dims_usize
            ));

        assert_eq!(
            broadcast_dims, dst_dims_usize,
            "GPU Add: Broadcast shape {:?} != dst shape {:?}",
            broadcast_dims, dst_dims_usize
        );

        let strides_a_usize = TensorDesc::broadcast_strides(&src1_dims_usize, &dst_dims_usize);
        let strides_b_usize = TensorDesc::broadcast_strides(&src2_dims_usize, &dst_dims_usize);

        let mut strides_a_arr = [0u32; 8];
        // Ensure strides_a_usize rank matches dst_dims_usize rank for consistency in shader
        // TensorDesc::broadcast_strides returns strides for the broadcasted shape (dst_dims_usize)
        for i in 0..strides_a_usize.len() {
            if i < 8 {
                // Defensive check, should match rank
                strides_a_arr[i] = strides_a_usize[i] as u32;
            }
        }

        let mut strides_b_arr = [0u32; 8];
        for i in 0..strides_b_usize.len() {
            if i < 8 {
                // Defensive check
                strides_b_arr[i] = strides_b_usize[i] as u32;
            }
        }

        // If rank is 0 (scalar), shader expects total=1. dims_arr will be [0,0,...]
        // Shader's total calculation: `for (uint i = 0; i < pc.rank; ++i) total *= pc.dims[i];`
        // If rank is 0, total = 1. If rank > 0 and a dim is 0, total = 0.
        // This seems fine. For rank 0, dims_arr elements are 0, but pc.rank is 0, so loop for total is skipped.
        // For safety, if rank > 0, ensure no dim is 0 unless intended.
        // However, dst_mem.size would be 0 if a dim is 0, leading to num_elements = 0.
        // The current num_elements calculation is fine.

        let push_const_values = PushConstants {
            rank,
            pad: 0, // Padding value, 0 is fine
            dims: dims_arr,
            strides_a: strides_a_arr,
            strides_b: strides_b_arr,
        };

        let push_constant_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &push_const_values as *const _ as *const u8,
                std::mem::size_of::<PushConstants>(),
            )
        };

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
                .get_pipeline(GPUMemoryOperation::Addition)
                .ok_or(format!(
                    "{:?} pipeline not found",
                    GPUMemoryOperation::Addition
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

            // Push constants to the shader
            gpu.get_device().cmd_push_constants(
                command_buffer,
                gpu.get_compute_pipelines().get_layout(), // Pipeline layout
                vk::ShaderStageFlags::COMPUTE,            // Shader stage
                0,                                        // Offset
                push_constant_bytes,                      // Data
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

    fn execute_cpu(&self, tensor_graph: &mut TensorGraph) {
        assert!(
            self.src1 != self.dst && self.src2 != self.dst,
            "Cannot use Add for in-place operation"
        );

        let src1 = &tensor_graph.tensors[self.src1];
        let src2 = &tensor_graph.tensors[self.src2];
        let dst = &tensor_graph.tensors[self.dst];

        let a = src1.desc.to_dims();
        let b = src2.desc.to_dims();
        let c = dst.desc.to_dims();

        let bc = TensorDesc::broadcast_shape(&a, &b)
            .expect(&format!("Can't broadcast {:?} vs {:?}", a, b));

        assert_eq!(bc, c, "Broadcast {:?} != dst {:?}", bc, c);

        let sa = TensorDesc::broadcast_strides(&a, &c);
        let sb = TensorDesc::broadcast_strides(&b, &c);

        let mut dd = dst
            .data
            .borrow_mut_cpu_data()
            .expect("Destination tensor should have CPU data");
        let d1 = src1
            .data
            .borrow_cpu_data()
            .expect("Source tensor 1 should have CPU data");
        let d2 = src2
            .data
            .borrow_cpu_data()
            .expect("Source tensor 2 should have CPU data");

        for i in 0..dd.len() {
            let idxs = TensorDesc::unravel(i, &c);
            let off1 = TensorDesc::offset(&idxs, &sa);
            let off2 = TensorDesc::offset(&idxs, &sb);
            dd[i] = d1[off1] + d2[off2];
        }
    }
}
