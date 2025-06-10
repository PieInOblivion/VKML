use crate::{
    dataloader::error::VKMLEngineError,
    gpu::vk_gpu::GPU,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

use super::instruction::Instruction;

#[derive(Clone)]
pub struct ConcatInstruction {
    pub sources: Vec<TensorId>,
    pub dst: TensorId,
    pub dim: usize,
}

impl Debug for ConcatInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Concat(sources={:?}, dst={}, dim={})",
            self.sources, self.dst, self.dim
        )
    }
}

impl Instruction for ConcatInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        self.sources.clone()
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if !new_inputs.is_empty() {
            self.sources = new_inputs.to_vec();
        }

        if !new_outputs.is_empty() {
            self.dst = new_outputs[0];
        }
    }

    fn create_command_buffer(
        &self,
        _gpu: &GPU,
        _command_buffer: vk::CommandBuffer,
        _tensor_graph: &TensorGraph,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Complex operation that would require custom shaders
        Err("GPU implementation of Concat not yet supported".into())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, tensor_graph: &mut TensorGraph) -> Result<(), VKMLEngineError> {
        let mut dst_data = tensor_graph.tensors[self.dst].data.borrow_mut_cpu_data()?;

        // Check if this is a supported concat dimension
        if self.dim != 1 {
            return Err(VKMLEngineError::VulkanLoadError(format!(
                "CPU Concat only implemented for dimension 1, got {}",
                self.dim
            )));
        }

        // Get the first source tensor to determine batch size
        let first_source = self.sources.first().ok_or_else(|| {
            VKMLEngineError::VulkanLoadError(
                "Concat requires at least one source tensor".to_string(),
            )
        })?;

        let src_tensor = &tensor_graph.tensors[*first_source];
        let src_dims = src_tensor.desc.to_dims();

        if src_dims.len() != 2 {
            return Err(VKMLEngineError::VulkanLoadError(
                "Concat only supports 2D tensors".to_string(),
            ));
        }

        let batch_size = src_dims[0];

        // Calculate total output size and verify all sources
        let mut total_features = 0;
        for &src_id in &self.sources {
            let src_tensor = &tensor_graph.tensors[src_id];
            let src_dims = src_tensor.desc.to_dims();

            if src_dims.len() != 2 {
                return Err(VKMLEngineError::VulkanLoadError(
                    "All source tensors must be 2D for Concat".to_string(),
                ));
            }

            if src_dims[0] != batch_size {
                return Err(VKMLEngineError::ShapeMismatch(format!(
                    "All source tensors must have same batch size {}, got {}",
                    batch_size, src_dims[0]
                )));
            }

            total_features += src_dims[1];
        }

        // Verify destination tensor size
        let dst_dims = tensor_graph.tensors[self.dst].desc.to_dims();
        if dst_dims.len() != 2 || dst_dims[0] != batch_size || dst_dims[1] != total_features {
            return Err(VKMLEngineError::ShapeMismatch(format!(
                "Destination shape {:?} doesn't match expected [{}x{}]",
                dst_dims, batch_size, total_features
            )));
        }

        // Perform concatenation
        let mut dst_idx = 0;

        // Concat along feature dimension
        for b in 0..batch_size {
            for &src_id in &self.sources {
                let src_data = tensor_graph.tensors[src_id].data.borrow_cpu_data()?;
                let src_tensor = &tensor_graph.tensors[src_id];
                let src_dims = src_tensor.desc.to_dims();
                let feat_dim = src_dims[1];

                let src_offset = b * feat_dim;

                // Copy this batch's features directly
                for i in 0..feat_dim {
                    dst_data[dst_idx] = src_data[src_offset + i];
                    dst_idx += 1;
                }
            }
        }

        Ok(())
    }
}
