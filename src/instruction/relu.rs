use crate::{
    dataloader::error::VKMLEngineError,
    gpu::vk_gpu::GPU,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use ash::vk;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use super::instruction::Instruction;

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
        gpu: &GPU,
        command_buffer: vk::CommandBuffer,
        tensor_graph: &TensorGraph,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let src_mem = tensor_graph.get_gpu_memory_or_panic(&self.src);
        let dst_mem = tensor_graph.get_gpu_memory_or_panic(&self.dst);

        gpu.create_relu_command_buffer(command_buffer, src_mem, dst_mem)
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, tensor_graph: &mut TensorGraph) -> Result<(), VKMLEngineError> {
        let src_data = tensor_graph.tensors[self.src].data.borrow_cpu_data()?;
        let mut dst_data = tensor_graph.tensors[self.dst].data.borrow_mut_cpu_data()?;

        // Verify tensor sizes
        if dst_data.len() != src_data.len() {
            return Err(VKMLEngineError::ShapeMismatch(format!(
                "Destination tensor size {} doesn't match source tensor size {}",
                dst_data.len(),
                src_data.len()
            )));
        }

        // Update in-place
        for i in 0..src_data.len() {
            dst_data[i] = src_data[i].max(0.0);
        }

        Ok(())
    }
}
