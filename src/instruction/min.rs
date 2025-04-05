use crate::{
    dataloader::error::VKMLEngineError,
    gpu::vk_gpu::GPU,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use ash::vk;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use super::instruction::Instruction;

#[derive(Clone)]
pub struct MinInstruction {
    pub src1: TensorId,
    pub src2: TensorId,
    pub dst: TensorId,
}

impl Debug for MinInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Min(src1={}, src2={}, dst={})",
            self.src1, self.src2, self.dst
        )
    }
}

impl Instruction for MinInstruction {
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

        gpu.create_min_command_buffer(command_buffer, src1_mem, src2_mem, dst_mem)
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, tensor_graph: &mut TensorGraph) -> Result<(), VKMLEngineError> {
        let src1_data = tensor_graph.tensors[self.src1].data.borrow_cpu_data()?;
        let src2_data = tensor_graph.tensors[self.src2].data.borrow_cpu_data()?;

        // Verify tensor sizes
        if src1_data.len() != src2_data.len() {
            return Err(VKMLEngineError::ShapeMismatch(format!(
                "Source tensors must have the same size: {} vs {}",
                src1_data.len(),
                src2_data.len()
            )));
        }

        let mut dst_data = tensor_graph.tensors[self.dst].data.borrow_mut_cpu_data()?;

        if dst_data.len() != src1_data.len() {
            return Err(VKMLEngineError::ShapeMismatch(format!(
                "Destination tensor size {} doesn't match source tensor size {}",
                dst_data.len(),
                src1_data.len()
            )));
        }

        // Update in-place
        for i in 0..src1_data.len() {
            dst_data[i] = src1_data[i].min(src2_data[i]);
        }

        Ok(())
    }
}
