use crate::{
    gpu::vk_gpu::GPU,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use std::{fmt::{Debug, Formatter, Result as FmtResult}, sync::Arc};
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
        _tensor_graph: &mut TensorGraph,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Complex operation that would require custom shaders
        Err("GPU implementation of Concat not yet supported".into())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, tensor_graph: Arc<TensorGraph>) {
        let mut dst_data = tensor_graph.tensors[self.dst].data.write_data();

        assert_eq!(
            self.dim, 1,
            "CPU Concat only implemented for dimension 1, got {}",
            self.dim
        );
        assert!(
            !self.sources.is_empty(),
            "Concat requires at least one source tensor"
        );

        let first_source = self.sources[0];
        let src_tensor = &tensor_graph.tensors[first_source];
        let src_dims = src_tensor.desc.to_dims();

        assert_eq!(src_dims.len(), 2, "Concat only supports 2D tensors");
        let batch_size = src_dims[0];

        let mut total_features = 0;
        for &src_id in &self.sources {
            let src_tensor = &tensor_graph.tensors[src_id];
            let src_dims = src_tensor.desc.to_dims();

            assert_eq!(
                src_dims.len(),
                2,
                "All source tensors must be 2D for Concat"
            );
            assert_eq!(
                src_dims[0], batch_size,
                "All source tensors must have same batch size {}, got {}",
                batch_size, src_dims[0]
            );

            total_features += src_dims[1];
        }

        let dst_dims = tensor_graph.tensors[self.dst].desc.to_dims();
        assert_eq!(dst_dims.len(), 2, "Destination tensor must be 2D");
        assert_eq!(dst_dims[0], batch_size, "Destination batch size mismatch");
        assert_eq!(
            dst_dims[1], total_features,
            "Destination feature size mismatch"
        );

        let mut dst_idx = 0;
        for b in 0..batch_size {
            for &src_id in &self.sources {
                let src_data = tensor_graph.tensors[src_id].data.read_data();
                let src_tensor = &tensor_graph.tensors[src_id];
                let src_dims = src_tensor.desc.to_dims();
                let feat_dim = src_dims[1];
                let src_offset = b * feat_dim;

                for i in 0..feat_dim {
                    dst_data[dst_idx] = src_data[src_offset + i];
                    dst_idx += 1;
                }
            }
        }
    }
}
