use crate::{
    dataloader::error::VKMLEngineError,
    gpu::vk_gpu::GPU,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use ash::vk;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use super::instruction::Instruction;

#[derive(Clone)]
pub struct SoftmaxInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    pub dim: usize,
}

impl Debug for SoftmaxInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Softmax(src={}, dst={}, dim={})",
            self.src, self.dst, self.dim
        )
    }
}

impl Instruction for SoftmaxInstruction {
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
        let tensor = tensor_graph.tensors.get(*&self.src).unwrap();

        gpu.create_softmax_command_buffer(
            command_buffer,
            src_mem,
            dst_mem,
            self.dim,
            &tensor.desc.to_dims(),
        )
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

        let tensor = &tensor_graph.tensors[self.src];
        let dims = tensor.desc.to_dims();

        // CPU implementation currently only supports softmax on the last dimension
        if self.dim != dims.len() - 1 {
            return Err(VKMLEngineError::VulkanLoadError(
                "CPU Softmax currently only supports the last dimension".to_string(),
            ));
        }

        let feature_size = dims[self.dim];
        let batch_size = src_data.len() / feature_size;

        // Process each batch separately
        for b in 0..batch_size {
            let offset = b * feature_size;

            // Find max for numerical stability
            let mut max_val = f32::MIN;
            for i in 0..feature_size {
                max_val = max_val.max(src_data[offset + i]);
            }

            // Compute exponentials and sum
            let mut sum = 0.0;
            for i in 0..feature_size {
                let exp_val = (src_data[offset + i] - max_val).exp();
                dst_data[offset + i] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for i in 0..feature_size {
                dst_data[offset + i] /= sum;
            }
        }

        Ok(())
    }
}
