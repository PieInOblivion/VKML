use crate::ComputeManager;
use crate::instruction::div::push_constants::DivPushConstants;
use crate::utils::as_bytes;
use crate::{
    gpu::vk_gpu::Gpu,
    instruction::{
        div::f32_cpu::f32_cpu, gpu_operations::GPUMemoryOperation, instruction::Instruction,
    },
    tensor::desc::TensorDesc,
    tensor_graph::tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk::Handle;
use vulkanalia::vk::KhrPushDescriptorExtension;
use vulkanalia::{vk, vk::DeviceV1_0};

#[derive(Clone)]
pub struct DivInstruction {
    pub src1: TensorId,
    pub src2: TensorId,
    pub dst: TensorId,
}

impl Debug for DivInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Div(src1={}, src2={}, dst={})",
            self.src1, self.src2, self.dst
        )
    }
}

impl Instruction for DivInstruction {
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
        gpu: &Gpu,
        command_buffer: vk::CommandBuffer,
        cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let src1_tensor = cm.tensor_read(self.src1);
        let src1_mem = src1_tensor.get_gpu_memory_or_panic();
        let src2_tensor = cm.tensor_read(self.src2);
        let src2_mem = src2_tensor.get_gpu_memory_or_panic();
        let dst_tensor = cm.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        // Get tensor descriptions for calculating broadcast shapes and strides
        let src1_desc = &src1_tensor.desc;
        let src2_desc = &src2_tensor.desc;
        let dst_desc = &dst_tensor.desc;

        let src1_dims_usize = src1_desc.to_dims();
        let src2_dims_usize = src2_desc.to_dims();
        let dst_dims_usize = dst_desc.to_dims();

        // Prepare push constant data using shared PushConstants

        let rank = dst_dims_usize.len() as u32;
        assert!(
            rank <= 8,
            "Div: tensor rank {} exceeds maximum supported rank of 8",
            rank
        );

        let mut dims_arr = [0u32; 8];
        for i in 0..dst_dims_usize.len() {
            dims_arr[i] = dst_dims_usize[i] as u32;
        }

        let broadcast_dims = TensorDesc::broadcast_shape(&src1_dims_usize, &src2_dims_usize)
            .expect(&format!(
                "GPU Div: Can't broadcast {:?} vs {:?}",
                src1_dims_usize, src2_dims_usize
            ));

        assert_eq!(
            broadcast_dims, dst_dims_usize,
            "GPU Div: Broadcast shape {:?} != dst shape {:?}",
            broadcast_dims, dst_dims_usize
        );

        let strides_a_usize = TensorDesc::broadcast_strides(&src1_dims_usize, &dst_dims_usize);
        let strides_b_usize = TensorDesc::broadcast_strides(&src2_dims_usize, &dst_dims_usize);

        let mut strides_a_arr = [0u32; 8];
        for i in 0..strides_a_usize.len() {
            if i < 8 {
                strides_a_arr[i] = strides_a_usize[i] as u32;
            }
        }

        let mut strides_b_arr = [0u32; 8];
        for i in 0..strides_b_usize.len() {
            if i < 8 {
                strides_b_arr[i] = strides_b_usize[i] as u32;
            }
        }

        let total_elements: u64 = dst_dims_usize.iter().map(|d| *d as u64).product();

        let push_const_values = DivPushConstants {
            rank,
            pad: 0,
            total: total_elements as u32,
            dims: dims_arr,
            strides_a: strides_a_arr,
            strides_b: strides_b_arr,
        };

        let push_constant_bytes = as_bytes(&push_const_values);

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default();

            gpu.get_device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            let buffer_infos = [
                vk::DescriptorBufferInfo {
                    buffer: src1_mem.buffer,
                    offset: 0,
                    range: src1_mem.size,
                },
                vk::DescriptorBufferInfo {
                    buffer: src2_mem.buffer,
                    offset: 0,
                    range: src2_mem.size,
                },
                vk::DescriptorBufferInfo {
                    buffer: dst_mem.buffer,
                    offset: 0,
                    range: dst_mem.size,
                },
            ];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: vk::DescriptorSet::null(),
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[0],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: vk::DescriptorSet::null(),
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[1],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    next: std::ptr::null(),
                    dst_set: vk::DescriptorSet::null(),
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[2],
                    image_info: std::ptr::null(),
                    texel_buffer_view: std::ptr::null(),
                },
            ];

            // Choose operation and element size based on tensor DataType
            let op_datatype = dst_tensor.desc.data_type();
            let gpu_op = match op_datatype {
                DataType::Float => GPUMemoryOperation::Divide_F32,
                _ => {
                    return Err(
                        format!("GPU Div unimplemented for DataType {:?}", op_datatype).into(),
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
                &write_descriptor_sets,
            );

            // Push constants to the shader
            gpu.get_device().cmd_push_constants(
                command_buffer,
                gpu.get_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constant_bytes,
            );

            let workgroup_size = 256;
            let num_elements: u64 = dst_dims_usize.iter().map(|d| *d as u64).product();
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

    fn execute_cpu(&self, cm: &ComputeManager) {
        // First check that this isn't being used as an in-place operation
        assert!(
            self.src1 != self.dst && self.src2 != self.dst,
            "Cannot use Div for in-place operation. Use DivInplace instead."
        );
        let src1_tensor = cm.tensor_read(self.src1);
        let src2_tensor = cm.tensor_read(self.src2);
        let dst_tensor = cm.tensor_write(self.dst);

        let a = src1_tensor.desc.to_dims();
        let b = src2_tensor.desc.to_dims();
        let c = dst_tensor.desc.to_dims();

        // 1) compute broadcast shape
        let bc = TensorDesc::broadcast_shape(&a, &b)
            .expect(&format!("Can't broadcast {:?} vs {:?}", a, b));
        assert_eq!(bc, c, "Broadcast {:?} != dst {:?}", bc, c);

        let sa = TensorDesc::broadcast_strides(&a, &c);
        let sb = TensorDesc::broadcast_strides(&b, &c);

        let op_datatype = dst_tensor.desc.data_type();

        let src1_bytes = src1_tensor.get_cpu_memory_slice_or_panic();
        let src2_bytes = src2_tensor.get_cpu_memory_slice_or_panic();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match op_datatype {
            DataType::Float => f32_cpu(sa, sb, c, src1_bytes, src2_bytes, dst_ptr),
            _ => unimplemented!(
                "div.rs unimplemented cpu instruction for DataType {:?}",
                dst_tensor.desc.data_type()
            ),
        }
    }
}
