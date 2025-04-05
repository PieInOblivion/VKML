use crate::{
    dataloader::error::VKMLEngineError,
    gpu::vk_gpu::GPU,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use ash::vk;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use super::instruction::Instruction;

#[derive(Clone)]
pub struct GELUInstruction {
    pub src: TensorId,
    pub dst: TensorId,
}

impl Debug for GELUInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "GELU(src={}, dst={})", self.src, self.dst)
    }
}

impl Instruction for GELUInstruction {
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

        gpu.create_gelu_command_buffer(command_buffer, src_mem, dst_mem)
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

        // GELU constants
        // GELU approximation: x * 0.5 * (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x^3)))
        let sqrt2_over_pi = 0.7978845608028654;
        let coef = 0.044715;

        // Update in-place
        for i in 0..src_data.len() {
            let val = src_data[i];
            dst_data[i] = val * 0.5 * (1.0 + (sqrt2_over_pi * (val + coef * val.powi(3))).tanh());
        }

        Ok(())
    }
}
