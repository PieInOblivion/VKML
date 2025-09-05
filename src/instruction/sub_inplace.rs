use crate::{
    gpu::{compute_pipelines::GPUMemoryOperation, vk_gpu::GPU},
    tensor::desc::TensorDesc,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use std::{fmt::{Debug, Formatter, Result as FmtResult}, sync::Arc};
use vulkanalia::{vk, vk::DeviceV1_0};

use super::instruction::Instruction;

#[derive(Clone)]
pub struct SubInplaceInstruction {
    pub dst: TensorId,
    pub src1: TensorId,
}

impl Debug for SubInplaceInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "SubInplace(dst={}, src1={})", self.dst, self.src1)
    }
}

impl Instruction for SubInplaceInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst, self.src1]
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if new_inputs.len() >= 2 {
            self.dst = new_inputs[0];
            self.src1 = new_inputs[1];
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
        let mem_a = tensor_graph.get_gpu_memory_or_panic(self.dst);
        let mem_b = tensor_graph.get_gpu_memory_or_panic(self.src1);

        let desc_a = &tensor_graph.tensors[self.dst].desc;
        let desc_b = &tensor_graph.tensors[self.src1].desc;

        let dims_a = desc_a.to_dims();
        let dims_b = desc_b.to_dims();

        // push‐constant layout
        #[repr(C)]
        struct PC {
            rank: u32,
            pad: u32,
            dims: [u32; 8],
            strides_a: [u32; 8],
            strides_b: [u32; 8],
        }

        // broadcast checks
        let bc = TensorDesc::broadcast_shape(&dims_a, &dims_b).expect(&format!(
            "InplaceSub: can't broadcast {:?} vs {:?}",
            dims_a, dims_b
        ));

        assert_eq!(
            bc, dims_a,
            "InplaceSub: broadcast {:?} != out {:?}",
            bc, dims_a
        );

        let rank = dims_a.len() as u32;
        assert!(
            rank <= 8,
            "InplaceSub: tensor rank {} exceeds maximum supported rank of 8",
            rank
        );

        let mut dims_arr = [0u32; 8];
        for i in 0..dims_a.len() {
            dims_arr[i] = dims_a[i] as u32;
        }
        let sa = TensorDesc::broadcast_strides(&dims_a, &dims_a);
        let sb = TensorDesc::broadcast_strides(&dims_b, &dims_a);
        let mut strides_a = [0u32; 8];
        let mut strides_b = [0u32; 8];
        for i in 0..sa.len() {
            strides_a[i] = sa[i] as u32;
            strides_b[i] = sb[i] as u32;
        }

        let pc_data = PC {
            rank,
            pad: 0,
            dims: dims_arr,
            strides_a,
            strides_b,
        };
        let pc_bytes = unsafe {
            std::slice::from_raw_parts(&pc_data as *const _ as *const u8, std::mem::size_of::<PC>())
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
            let ds = gpu.get_device().allocate_descriptor_sets(&alloc_info)?[0];

            let infos = [
                vk::DescriptorBufferInfo {
                    buffer: mem_a.buffer,
                    offset: 0,
                    range: mem_a.size,
                },
                vk::DescriptorBufferInfo {
                    buffer: mem_b.buffer,
                    offset: 0,
                    range: mem_b.size,
                },
            ];
            let writes = [
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: ds,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &infos[0],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: ds,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &infos[1],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
            ];
            gpu.get_device()
                .update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);

            let pipeline = gpu
                .get_compute_pipelines()
                .get_pipeline(GPUMemoryOperation::SubtractInplace)
                .ok_or(format!("SubtractInplace pipeline not found"))?;

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
                &[ds],
                &[],
            );
            gpu.get_device().cmd_push_constants(
                command_buffer,
                gpu.get_compute_pipelines().get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            let wg = 256;
            let n = mem_a.size / std::mem::size_of::<f32>() as u64;
            let groups = ((n + wg as u64 - 1) / wg as u64) as u32;
            gpu.get_device().cmd_dispatch(command_buffer, groups, 1, 1);
            gpu.get_device().end_command_buffer(command_buffer)?;
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, tensor_graph: Arc<TensorGraph>) {
        let a = &tensor_graph.tensors[self.dst];
        let b = &tensor_graph.tensors[self.src1];
        let da = a.desc.to_dims();
        let db = b.desc.to_dims();

        let out = TensorDesc::broadcast_shape(&da, &db)
            .expect(&format!("Can't broadcast {:?} vs {:?}", da, db));
        let mut data_a = a.data.write_data();
        let data_b = b.data.read_data();

        let sa = TensorDesc::broadcast_strides(&da, &out);
        let sb = TensorDesc::broadcast_strides(&db, &out);

        for i in 0..data_a.len() {
            let idxs = TensorDesc::unravel(i, &out);
            let offa = TensorDesc::offset(&idxs, &sa);
            let offb = TensorDesc::offset(&idxs, &sb);
            data_a[offa] -= data_b[offb];
        }
    }
}
