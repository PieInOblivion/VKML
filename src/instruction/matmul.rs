use crate::{
    dataloader::error::VKMLEngineError,
    gpu::vk_gpu::GPU,
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use ash::vk;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use super::instruction::Instruction;

#[derive(Clone)]
pub struct MatMulInstruction {
    pub src1: TensorId,
    pub src2: TensorId,
    pub dst: TensorId,
}

impl Debug for MatMulInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "MatMul(src1={}, src2={}, dst={})",
            self.src1, self.src2, self.dst
        )
    }
}

impl Instruction for MatMulInstruction {
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
        let src1_tensor = tensor_graph.tensors.get(*&self.src1).unwrap();
        let src2_tensor = tensor_graph.tensors.get(*&self.src2).unwrap();
        let dst_tensor = tensor_graph.tensors.get(*&self.dst).unwrap();

        gpu.create_matmul_command_buffer(command_buffer, src1_tensor, src2_tensor, dst_tensor)
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, tensor_graph: &mut TensorGraph) -> Result<(), VKMLEngineError> {
        let src1_data = tensor_graph.tensors[self.src1].data.borrow_cpu_data()?;
        let src2_data = tensor_graph.tensors[self.src2].data.borrow_cpu_data()?;
        let mut dst_data = tensor_graph.tensors[self.dst].data.borrow_mut_cpu_data()?;

        let src1_tensor = tensor_graph.tensors.get(self.src1).unwrap();
        let src2_tensor = tensor_graph.tensors.get(self.src2).unwrap();
        let dst_tensor = tensor_graph.tensors.get(self.dst).unwrap();

        let src1_dims = src1_tensor.desc.to_dims();
        let src2_dims = src2_tensor.desc.to_dims();
        let dst_dims = dst_tensor.desc.to_dims();

        // Zero initialize result
        for val in dst_data.iter_mut() {
            *val = 0.0;
        }

        // Handle special cases for 1D tensors
        let (effective_src1_dims, effective_src2_dims) = match (src1_dims.len(), src2_dims.len()) {
            (1, 1) => {
                return Err(VKMLEngineError::VulkanLoadError(
                    "MatMul between two 1D tensors is not supported".to_string(),
                ));
            }
            (1, _) => {
                // Convert [k] to [1,k] for matrix multiplication purposes
                let mut dims = Vec::with_capacity(src1_dims.len() + 1);
                dims.push(1);
                dims.extend_from_slice(&src1_dims);
                (dims, src2_dims.clone())
            }
            (_, 1) => {
                // Convert [k] to [k,1] for matrix multiplication purposes
                let mut dims = Vec::with_capacity(src2_dims.len() + 1);
                dims.extend_from_slice(&src2_dims);
                dims.push(1);
                (src1_dims.clone(), dims)
            }
            _ => (src1_dims.clone(), src2_dims.clone()),
        };

        // Extract core matrix dimensions
        if effective_src1_dims.len() < 2 || effective_src2_dims.len() < 2 {
            return Err(VKMLEngineError::VulkanLoadError(
                "After adjustment, tensors must have at least 2 dimensions for MatMul".to_string(),
            ));
        }

        let src1_matrix_dims = &effective_src1_dims[effective_src1_dims.len() - 2..];
        let src2_matrix_dims = &effective_src2_dims[effective_src2_dims.len() - 2..];

        // Check inner dimensions match
        let m = src1_matrix_dims[0];
        let k1 = src1_matrix_dims[1];
        let k2 = src2_matrix_dims[0];
        let n = src2_matrix_dims[1];

        if k1 != k2 {
            return Err(VKMLEngineError::ShapeMismatch(format!(
                "Inner dimensions don't match for matrix multiplication: {} vs {}",
                k1, k2
            )));
        }

        // Extract batch dimensions
        let src1_batch_dims = &effective_src1_dims[..effective_src1_dims.len() - 2];
        let src2_batch_dims = &effective_src2_dims[..effective_src2_dims.len() - 2];

        // Validate batch dimensions compatibility (must be either identical or one is empty)
        let batch_dims = if src1_batch_dims.is_empty() {
            src2_batch_dims.to_vec()
        } else if src2_batch_dims.is_empty() {
            src1_batch_dims.to_vec()
        } else if src1_batch_dims == src2_batch_dims {
            src1_batch_dims.to_vec()
        } else {
            return Err(VKMLEngineError::ShapeMismatch(format!(
                "Incompatible batch dimensions: {:?} vs {:?}",
                src1_batch_dims, src2_batch_dims
            )));
        };

        // Calculate expected output dims
        let mut expected_output_dims = batch_dims.clone();
        expected_output_dims.push(m);
        expected_output_dims.push(n);

        // Handle the case of 1D output for vector-vector result
        let expected_output_dims = match (src1_dims.len(), src2_dims.len()) {
            (1, _) => {
                // For [k] × [batch,k,n] → [batch,n]
                expected_output_dims[expected_output_dims.len() - 2..].to_vec()
            }
            (_, 1) => {
                // For [batch,m,k] × [k] → [batch,m]
                let mut dims = batch_dims.clone();
                dims.push(m);
                dims
            }
            _ => expected_output_dims,
        };

        // Validate output dimensions
        if dst_dims != expected_output_dims {
            return Err(VKMLEngineError::ShapeMismatch(format!(
                "Output dimensions mismatch: expected {:?}, got {:?}",
                expected_output_dims, dst_dims
            )));
        }

        // Calculate strides for each tensor
        let calculate_strides = |dims: &[usize]| -> Vec<usize> {
            let mut strides = vec![1; dims.len()];
            let mut stride = 1;
            for i in (0..dims.len()).rev() {
                strides[i] = stride;
                stride *= dims[i];
            }
            strides
        };

        let src1_strides = calculate_strides(&effective_src1_dims);
        let src2_strides = calculate_strides(&effective_src2_dims);
        let dst_strides = calculate_strides(&dst_dims);

        // Calculate total number of batches
        let total_batches = batch_dims.iter().product::<usize>().max(1);

        // Get multi-dimensional indices from flat index
        let get_indices = |flat_idx: usize, dims: &[usize]| -> Vec<usize> {
            let mut indices = vec![0; dims.len()];
            let mut remaining = flat_idx;

            for i in (0..dims.len()).rev() {
                indices[i] = remaining % dims[i];
                remaining /= dims[i];
            }

            indices
        };

        // Calculate offset from indices and strides
        let calculate_offset = |indices: &[usize], strides: &[usize]| -> usize {
            let mut offset = 0;
            for (i, &idx) in indices.iter().enumerate() {
                offset += idx * strides[i];
            }
            offset
        };

        // Execute batched matrix multiplication
        for batch_idx in 0..total_batches {
            // Calculate batch indices
            let batch_indices = if !batch_dims.is_empty() {
                get_indices(batch_idx, &batch_dims)
            } else {
                vec![]
            };

            // Calculate batch offsets
            let src1_batch_offset = if src1_batch_dims.is_empty() {
                0
            } else {
                calculate_offset(&batch_indices, &src1_strides[..src1_batch_dims.len()])
            };

            let src2_batch_offset = if src2_batch_dims.is_empty() {
                0
            } else {
                calculate_offset(&batch_indices, &src2_strides[..src2_batch_dims.len()])
            };

            let dst_batch_offset = if batch_dims.is_empty() {
                0
            } else {
                let effective_dst_batch_dims = dst_dims.len() - 2;
                calculate_offset(&batch_indices, &dst_strides[..effective_dst_batch_dims])
            };

            // Matrix multiplication for this batch
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;

                    for kk in 0..k1 {
                        // Calculate indices with proper striding
                        let src1_idx = src1_batch_offset
                            + i * src1_strides[effective_src1_dims.len() - 2]
                            + kk * src1_strides[effective_src1_dims.len() - 1];

                        let src2_idx = src2_batch_offset
                            + kk * src2_strides[effective_src2_dims.len() - 2]
                            + j * src2_strides[effective_src2_dims.len() - 1];

                        // Handle 1D special cases
                        let src1_val = if src1_dims.len() == 1 {
                            src1_data[kk]
                        } else {
                            src1_data[src1_idx]
                        };

                        let src2_val = if src2_dims.len() == 1 {
                            src2_data[kk]
                        } else {
                            src2_data[src2_idx]
                        };

                        sum += src1_val * src2_val;
                    }

                    // Handle result placement based on output dimensionality
                    let dst_idx = if src1_dims.len() == 1 && dst_dims.len() == 1 {
                        // [k] × [k,n] → [n]
                        j
                    } else if src2_dims.len() == 1 && dst_dims.len() == 1 {
                        // [m,k] × [k] → [m]
                        i
                    } else {
                        dst_batch_offset
                            + i * dst_strides[dst_dims.len() - 2]
                            + j * dst_strides[dst_dims.len() - 1]
                    };

                    dst_data[dst_idx] = sum;
                }
            }
        }

        Ok(())
    }
}
