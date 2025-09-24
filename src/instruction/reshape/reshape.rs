use crate::{
    gpu::vk_gpu::GPU,
    instruction::instruction::Instruction,
    tensor_graph::tensor_graph::TensorId, ComputeManager,
};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::{vk, vk::DeviceV1_0};

#[derive(Clone)]
pub struct ReshapeInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    // The target shape values as provided by the instruction (int64 values). May contain -1 and 0.
    pub shape_values: Vec<i64>,
    // allowzero forwarded from ONNX attribute
    pub allowzero: Option<i64>,
}

impl Debug for ReshapeInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Reshape(src={}, dst={}, shape={:?}, allowzero={:?})",
            self.src, self.dst, self.shape_values, self.allowzero
        )
    }
}

impl Instruction for ReshapeInstruction {
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
        // Reshape in Vulkan is a logical operation, not a physical one
        // We essentially need to copy data between the tensors
        let src_tensor = cm.tensor_read(self.src);
        let src_mem = src_tensor.get_gpu_memory_or_panic();
        let dst_tensor = cm.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                next: std::ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                inheritance_info: std::ptr::null(),
            };

            gpu.get_device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            // Copy regions - entire buffer
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: src_mem.size,
            };

            gpu.get_device().cmd_copy_buffer(
                command_buffer,
                src_mem.buffer,
                dst_mem.buffer,
                &[copy_region],
            );

            gpu.get_device().end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        // Start from the stored shape values provided by the instruction
        let mut new_dims = self.shape_values.clone();

        // Apply allowzero semantics: if allowzero is None or 0, zeros copy from input
        let allowzero_flag = self.allowzero.unwrap_or(0) != 0;

        if !allowzero_flag {
            let src_desc = cm.tensor_read(self.src).desc.clone();
            let src_dims = src_desc.to_dims();
            for (i, val) in new_dims.iter_mut().enumerate() {
                if *val == 0 {
                    *val = *src_dims.get(i).unwrap_or(&1);
                }
            }
        } else {
            // allowzero==1: mixing -1 and 0 is invalid per ONNX. Check and error.
            if new_dims.iter().any(|&d| d == 0) && new_dims.iter().any(|&d| d == -1) {
                panic!("Reshape: 'allowzero' set but shape contains both 0 and -1");
            }
        }

        // Infer -1 if present
        let src_desc = cm.tensor_read(self.src).desc.clone();
        let src_num = src_desc.num_elements();
        let neg1_count = new_dims.iter().filter(|&&d| d == -1).count();
        if neg1_count > 1 {
            panic!("Reshape: more than one -1 in shape is not allowed");
        }
        if neg1_count == 1 {
            let mut prod = 1usize;
            for &d in &new_dims {
                if d == -1 {
                    continue;
                }
                if d < 0 {
                    panic!("Reshape: negative dimensions other than -1 not allowed");
                }
                prod = prod.saturating_mul(d as usize);
            }
            if prod == 0 || src_num % prod != 0 {
                panic!("Reshape: cannot infer -1 dimension at runtime");
            }
            let inferred = (src_num / prod) as i64;
            for v in new_dims.iter_mut() {
                if *v == -1 {
                    *v = inferred;
                }
            }
        } else {
            // verify product matches
            let prod: usize = new_dims.iter().map(|&d| d as usize).product();
            if prod != src_num {
                panic!("Reshape: total elements do not match input tensor at runtime");
            }
        }

        // Update dst tensor descriptor (panic on failure to preserve current invariants)
        {
            let mut dst_t = cm.tensor_write(self.dst);
            dst_t
                .desc
                .reshape(new_dims)
                .expect("Invalid reshape at runtime");
        }

        // Copy data between underlying buffers (reshape is logical concerning layout)
        let src_tensor = cm.tensor_read(self.src);
        let mut dst_tensor = cm.tensor_write(self.dst);

        dst_tensor.write(&src_tensor.read());
    }
}
