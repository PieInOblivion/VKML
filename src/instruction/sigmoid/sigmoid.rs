use crate::{
    ComputeManager,
    gpu::vk_gpu::Gpu,
    instruction::{
        gpu_operations::GPUMemoryOperation, instruction::Instruction, sigmoid::f32_cpu::f32_cpu,
    },
    tensor::desc::TensorDesc,
    tensor_graph::tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::{
    fmt::{Debug, Formatter, Result as FmtResult},
    ptr,
};
use vulkanalia::vk::Handle;
use vulkanalia::vk::KhrPushDescriptorExtension;
use vulkanalia::{vk, vk::DeviceV1_0};

#[derive(Clone)]
pub struct SigmoidInstruction {
    pub src: TensorId,
    pub dst: TensorId,
}

impl Debug for SigmoidInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "Sigmoid(src={}, dst={})", self.src, self.dst)
    }
}

impl Instruction for SigmoidInstruction {
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
        gpu: &Gpu,
        command_buffer: vk::CommandBuffer,
        cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let src_tensor = cm.tensor_read(self.src);
        let src_mem = src_tensor.get_gpu_memory_or_panic();
        let dst_tensor = cm.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default();

            gpu.get_device()
                .begin_command_buffer(command_buffer, &begin_info)?;

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
                    dst_set: vk::DescriptorSet::null(),
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
                    dst_set: vk::DescriptorSet::null(),
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    buffer_info: &buffer_infos[1],
                    image_info: ptr::null(),
                    texel_buffer_view: ptr::null(),
                },
            ];

            let op_datatype = dst_tensor.desc.data_type();
            let gpu_op = match op_datatype {
                DataType::Float => GPUMemoryOperation::Sigmoid_F32,
                _ => {
                    return Err(format!(
                        "GPU Sigmoid unimplemented for DataType {:?}",
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

            gpu.get_device().cmd_push_descriptor_set_khr(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                gpu.get_layout(),
                0,
                &write_descriptor_sets,
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

    fn execute_cpu(&self, cm: &ComputeManager) {
        // Follow add.rs pattern: compute broadcast shapes/strides and delegate to helper
        assert!(
            self.src != self.dst,
            "Cannot use Sigmoid for in-place operation"
        );

        let src_tensor = cm.tensor_read(self.src);
        let dst_tensor = cm.tensor_write(self.dst);

        let a = src_tensor.desc.to_dims();
        let c = dst_tensor.desc.to_dims();

        let bc = TensorDesc::broadcast_shape(&a, &c)
            .expect(&format!("Can't broadcast {:?} vs {:?}", a, c));
        assert_eq!(bc, c, "Broadcast {:?} != dst {:?}", bc, c);

        let sa = TensorDesc::broadcast_strides(&a, &c);

        let op_datatype = dst_tensor.desc.data_type();

        let src_bytes = src_tensor.get_cpu_memory_slice_or_panic();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match op_datatype {
            DataType::Float => f32_cpu(sa, vec![], c, src_bytes, &[], dst_ptr),
            _ => unimplemented!(
                "sigmoid.rs unimplemented cpu instruction for DataType {:?}",
                dst_tensor.desc.data_type()
            ),
        }
    }
}
