use crate::{
    ComputeManager,
    gpu::vk_gpu::Gpu,
    instruction::{
        gpu_operations::GPUMemoryOperation, instruction::Instruction, relu::f32_cpu::f32_cpu,
    },
    tensor::desc::TensorDesc,
    tensor_graph::tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

#[derive(Clone)]
pub struct ReLUInstruction {
    pub src: TensorId,
    pub dst: TensorId,
}

impl Debug for ReLUInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "ReLU(src={}, dst={})", self.src, self.dst)
    }
}

impl Instruction for ReLUInstruction {
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

        gpu.begin_command_buffer(command_buffer)?;

        // Choose operation based on DataType (only Float supported)
        let op_datatype = dst_tensor.desc.data_type();
        let gpu_op = match op_datatype {
            DataType::Float => GPUMemoryOperation::ReLU_F32,
            _ => {
                return Err(
                    format!("GPU ReLU unimplemented for DataType {:?}", op_datatype).into(),
                );
            }
        };

        // Bind pipeline first so descriptor push is associated with the correct layout
        gpu.bind_compute_pipeline(command_buffer, gpu_op);

        gpu.bind_storage_buffers(command_buffer, &[&src_mem, &dst_mem]);

        let workgroup_size = 256;
        let num_elements = dst_mem.size / std::mem::size_of::<f32>() as u64;
        let num_workgroups = num_elements.div_ceil(workgroup_size as u64);

        gpu.dispatch(command_buffer, num_workgroups as u32, 1, 1);

        gpu.end_command_buffer(command_buffer)?;

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        // Follow add.rs style: compute broadcast shapes/strides and dispatch to typed helpers
        assert!(
            self.src != self.dst,
            "Cannot use ReLU for in-place operation"
        );

        let src_tensor = cm.tensor_read(self.src);
        let dst_tensor = cm.tensor_write(self.dst);

        let a = src_tensor.desc.dims();
        let c = dst_tensor.desc.dims().to_vec();

        let bc = TensorDesc::broadcast_shape(a, &c)
            .unwrap_or_else(|| panic!("Can't broadcast {:?} vs {:?}", a, c));
        assert_eq!(bc.as_slice(), c, "Broadcast {:?} != dst {:?}", bc, c);

        let sa = TensorDesc::broadcast_strides(a, &c);

        let op_datatype = dst_tensor.desc.data_type();

        let src_bytes = src_tensor.get_cpu_memory_slice_or_panic();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match op_datatype {
            DataType::Float => {
                f32_cpu(sa, c, src_bytes, dst_ptr);
            }
            _ => unimplemented!(
                "relu.rs unimplemented cpu instruction for DataType {:?}",
                dst_tensor.desc.data_type()
            ),
        }
    }
}
