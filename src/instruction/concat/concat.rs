use crate::{
    ComputeManager, gpu::vk_gpu::Gpu, instruction::instruction::Instruction,
    tensor_graph::tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

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
        _gpu: &Gpu,
        _command_buffer: vk::CommandBuffer,
        _cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Complex operation that would require custom shaders
        Err("GPU implementation of Concat not yet supported".into())
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        assert_eq!(
            self.dim, 1,
            "CPU Concat only implemented for dimension 1, got {}",
            self.dim
        );
        assert!(
            !self.sources.is_empty(),
            "Concat requires at least one source tensor"
        );

        // Prepare tensors and validate shapes
        let first_source = self.sources[0];
        let first_desc = cm.tensor_read(first_source).desc.to_dims();
        assert_eq!(first_desc.len(), 2, "Concat only supports 2D tensors");
        let batch_size = first_desc[0] as usize;

        let mut total_features: usize = 0;
        let mut source_features: Vec<usize> = Vec::with_capacity(self.sources.len());
        for &src_id in &self.sources {
            let src_desc = cm.tensor_read(src_id).desc.to_dims();
            assert_eq!(
                src_desc.len(),
                2,
                "All source tensors must be 2D for Concat"
            );
            assert_eq!(
                src_desc[0] as usize, batch_size,
                "All source tensors must have same batch size"
            );
            let feat = src_desc[1] as usize;
            source_features.push(feat);
            total_features += feat;
        }

        let dst_desc = cm.tensor_read(self.dst).desc.to_dims();
        assert_eq!(dst_desc.len(), 2, "Destination tensor must be 2D");
        assert_eq!(
            dst_desc[0] as usize, batch_size,
            "Destination batch size mismatch"
        );
        assert_eq!(
            dst_desc[1] as usize, total_features,
            "Destination feature size mismatch"
        );

        // Copy source CPU buffers into owned Vec<u8> first so we don't hold
        // immutable borrows on `tensor_graph` while we later need a mutable
        // borrow for the destination tensor. Also gather each source's dims.
        let mut owned_src_bytes: Vec<Vec<u8>> = Vec::with_capacity(self.sources.len());
        let mut src_dims_vec: Vec<Vec<i64>> = Vec::with_capacity(self.sources.len());
        for &src_id in &self.sources {
            let src_tensor = cm.tensor_read(src_id);
            let src_slice = src_tensor.get_cpu_memory_slice_or_panic();
            owned_src_bytes.push(src_slice.to_vec());
            src_dims_vec.push(src_tensor.desc.to_dims());
        }

        // Build a vector of byte-slice references that point into our owned buffers.
        let src_bytes_vec: Vec<&[u8]> = owned_src_bytes.iter().map(|v| v.as_slice()).collect();

        // Now it's safe to take a mutable borrow for the destination tensor.
        let dst_tensor = cm.tensor_write(self.dst);
        let op_datatype = dst_tensor.desc.data_type();
        match op_datatype {
            DataType::Float => {
                let dst_bytes = dst_tensor.get_cpu_memory_mut_slice_or_panic();
                crate::instruction::concat::f32_cpu::f32_cpu(
                    &src_bytes_vec,
                    &src_dims_vec,
                    self.dim,
                    &dst_desc,
                    dst_bytes,
                );
            }
            _ => unimplemented!(
                "concat.rs unimplemented cpu instruction for DataType {:?}",
                dst_tensor.desc.data_type()
            ),
        }
    }
}
