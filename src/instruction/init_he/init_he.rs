use crate::ComputeManager;
use crate::instruction::init_he::push_constants::InitHePushConstants;
use crate::utils::as_bytes;
use crate::{
    gpu::vk_gpu::Gpu,
    instruction::{
        gpu_operations::GPUOperation, init_he::f32_cpu::f32_cpu, instruction::Instruction,
    },
    tensor_graph::tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

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

        // Prepare push constants and CPU-side values
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

        // begin command buffer
        gpu.begin_command_buffer(command_buffer)?;

        // Choose operation based on data type
        let op_datatype = dst_tensor.desc.data_type();
        let gpu_op = match op_datatype {
            DataType::Float => GPUOperation::InitHe_F32,
            _ => {
                return Err(
                    format!("GPU InitHe unimplemented for DataType {:?}", op_datatype).into(),
                );
            }
        };

        // Choose and bind workgroup after we know total elements
        let local_size = gpu.optimal_workgroup_size_1d(dst_elems);
        gpu.bind_compute_pipeline(command_buffer, gpu_op, local_size);

        // Bind descriptor (dst at binding 0)
        gpu.bind_storage_buffers(command_buffer, &[&dst_mem]);

        // Push constants
        gpu.bind_push_constants(command_buffer, pc_bytes);

        // Dispatch
        gpu.dispatch(command_buffer, local_size, [dst_elems, 1, 1]);

        gpu.end_command_buffer(command_buffer)?;

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
