use crate::{
    dataloader::error::VKMLEngineError,
    gpu::{compute_pipelines::GPUMemoryOperation, vk_gpu::GPU},
    tensor::{compute_tensor::ComputeTensor, tensor_data::TensorData},
    tensor_graph::tensor_graph::{TensorGraph, TensorId},
};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::{vk, vk::DeviceV1_0};

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
        let src1_tensor = tensor_graph.tensors.get(self.src1).unwrap();
        let src2_tensor = tensor_graph.tensors.get(self.src2).unwrap();
        let dst_tensor = tensor_graph.tensors.get(self.dst).unwrap();

        let operation =
            determine_matmul_variant(&src1_tensor.desc.to_dims(), &src2_tensor.desc.to_dims());

        if operation == GPUMemoryOperation::MatMul {
            // Use generic fallback for unsupported dimension combinations
            create_generic_matmul_command_buffer(
                gpu,
                command_buffer,
                src1_tensor,
                src2_tensor,
                dst_tensor,
            )
        } else {
            // Use specialised implementation for supported dimensions
            create_specialized_matmul_command_buffer(
                gpu,
                command_buffer,
                src1_tensor,
                src2_tensor,
                dst_tensor,
                operation,
            )
        }
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, tensor_graph: &mut TensorGraph) -> Result<(), VKMLEngineError> {
        if self.src1 == self.dst || self.src2 == self.dst {
            return Err(VKMLEngineError::VulkanLoadError(
                "Cannot use MatMul for in-place operation".to_string(),
            ));
        }

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

fn determine_matmul_variant(src1_dims: &[usize], src2_dims: &[usize]) -> GPUMemoryOperation {
    match (src1_dims.len(), src2_dims.len()) {
        (1, 2) => GPUMemoryOperation::MatMul1D2D,
        (2, 1) => GPUMemoryOperation::MatMul2D1D,
        (2, 2) => GPUMemoryOperation::MatMul2D2D,
        (2, 3) => GPUMemoryOperation::MatMul2D3D,
        (3, 2) => GPUMemoryOperation::MatMul3D2D,
        (3, 3) => GPUMemoryOperation::MatMul3D3D,
        (3, 1) => GPUMemoryOperation::MatMul3D1D,
        (1, 3) => GPUMemoryOperation::MatMul1D3D,
        _ => {
            // Fallback to generic, or error
            eprintln!(
                "Unsupported tensor dimensions for matmul: {:?} x {:?}",
                src1_dims, src2_dims
            );
            GPUMemoryOperation::MatMul
        }
    }
}

fn configure_matmul_operation(
    operation: GPUMemoryOperation,
    src1_tensor: &ComputeTensor,
    src2_tensor: &ComputeTensor,
    dst_tensor: &ComputeTensor,
) -> Result<(Vec<u32>, u32, u32, u32), Box<dyn std::error::Error>> {
    let src1_dims = src1_tensor.desc.to_dims();
    let src2_dims = src2_tensor.desc.to_dims();

    let src1_strides = src1_tensor.desc.strides();
    let src2_strides = src2_tensor.desc.strides();
    let dst_strides = dst_tensor.desc.strides();

    match operation {
        GPUMemoryOperation::MatMul1D2D => {
            // [k] × [k,n] → [n]
            let k = src1_dims[0];
            let n = src2_dims[1];

            let push_constants = vec![
                k as u32,               // k
                n as u32,               // n
                src1_strides[0] as u32, // stride_a
                src2_strides[0] as u32, // stride_b0 (row stride)
                src2_strides[1] as u32, // stride_b1 (column stride)
                dst_strides[0] as u32,  // stride_c
            ];

            let workgroup_size = 256;
            let num_groups_x = (n as u32 + workgroup_size - 1) / workgroup_size;

            Ok((push_constants, num_groups_x, 1, 1))
        }

        GPUMemoryOperation::MatMul2D1D => {
            // [m,k] × [k] → [m]
            let m = src1_dims[0];
            let k = src1_dims[1];

            let push_constants = vec![
                m as u32,               // m
                k as u32,               // k
                src1_strides[0] as u32, // stride_a0 (row stride)
                src1_strides[1] as u32, // stride_a1 (column stride)
                src2_strides[0] as u32, // stride_b
                dst_strides[0] as u32,  // stride_c
            ];

            let workgroup_size = 256;
            let num_groups_x = (m as u32 + workgroup_size - 1) / workgroup_size;

            Ok((push_constants, num_groups_x, 1, 1))
        }

        GPUMemoryOperation::MatMul2D2D => {
            // [m,k] × [k,n] → [m,n]
            let m = src1_dims[0];
            let k = src1_dims[1];
            let n = src2_dims[1];

            let push_constants = vec![
                m as u32,               // m
                k as u32,               // k
                n as u32,               // n
                src1_strides[0] as u32, // stride_a0 (row stride)
                src1_strides[1] as u32, // stride_a1 (column stride)
                src2_strides[0] as u32, // stride_b0 (row stride)
                src2_strides[1] as u32, // stride_b1 (column stride)
                dst_strides[0] as u32,  // stride_c0 (row stride)
                dst_strides[1] as u32,  // stride_c1 (column stride)
            ];

            // Calculate workgroup dimensions - 16×16 threads per workgroup
            let workgroup_size = 16;
            let num_groups_x = (n as u32 + workgroup_size - 1) / workgroup_size;
            let num_groups_y = (m as u32 + workgroup_size - 1) / workgroup_size;

            Ok((push_constants, num_groups_x, num_groups_y, 1))
        }

        GPUMemoryOperation::MatMul2D3D => {
            // [m,k] × [batch,k,n] → [batch,m,n]
            let m = src1_dims[0];
            let k = src1_dims[1];
            let batch = src2_dims[0];
            let n = src2_dims[2];

            let push_constants = vec![
                batch as u32,           // batch
                m as u32,               // m
                k as u32,               // k
                n as u32,               // n
                src1_strides[0] as u32, // stride_a0 (row stride)
                src1_strides[1] as u32, // stride_a1 (column stride)
                src2_strides[0] as u32, // stride_b0 (batch stride)
                src2_strides[1] as u32, // stride_b1 (row stride)
                src2_strides[2] as u32, // stride_b2 (column stride)
                dst_strides[0] as u32,  // stride_c0 (batch stride)
                dst_strides[1] as u32,  // stride_c1 (row stride)
                dst_strides[2] as u32,  // stride_c2 (column stride)
            ];

            // Calculate workgroup dimensions - 8×8×4 threads per workgroup
            let workgroup_size_xy = 8;
            let workgroup_size_z = 4;
            let num_groups_x = (n as u32 + workgroup_size_xy - 1) / workgroup_size_xy;
            let num_groups_y = (m as u32 + workgroup_size_xy - 1) / workgroup_size_xy;
            let num_groups_z = (batch as u32 + workgroup_size_z - 1) / workgroup_size_z;

            Ok((push_constants, num_groups_x, num_groups_y, num_groups_z))
        }

        GPUMemoryOperation::MatMul3D2D => {
            // [batch,m,k] × [k,n] → [batch,m,n]
            let batch = src1_dims[0];
            let m = src1_dims[1];
            let k = src1_dims[2];
            let n = src2_dims[1];

            let push_constants = vec![
                batch as u32,           // batch
                m as u32,               // m
                k as u32,               // k
                n as u32,               // n
                src1_strides[0] as u32, // stride_a0 (batch stride)
                src1_strides[1] as u32, // stride_a1 (row stride)
                src1_strides[2] as u32, // stride_a2 (column stride)
                src2_strides[0] as u32, // stride_b0 (row stride)
                src2_strides[1] as u32, // stride_b1 (column stride)
                dst_strides[0] as u32,  // stride_c0 (batch stride)
                dst_strides[1] as u32,  // stride_c1 (row stride)
                dst_strides[2] as u32,  // stride_c2 (column stride)
            ];

            // Calculate workgroup dimensions - 8×8×4 threads per workgroup
            let workgroup_size_xy = 8;
            let workgroup_size_z = 4;
            let num_groups_x = (n as u32 + workgroup_size_xy - 1) / workgroup_size_xy;
            let num_groups_y = (m as u32 + workgroup_size_xy - 1) / workgroup_size_xy;
            let num_groups_z = (batch as u32 + workgroup_size_z - 1) / workgroup_size_z;

            Ok((push_constants, num_groups_x, num_groups_y, num_groups_z))
        }

        GPUMemoryOperation::MatMul3D3D => {
            // [batch,m,k] × [batch,k,n] → [batch,m,n]
            let batch = src1_dims[0];
            let m = src1_dims[1];
            let k = src1_dims[2];
            let n = src2_dims[2];

            let push_constants = vec![
                batch as u32,           // batch
                m as u32,               // m
                k as u32,               // k
                n as u32,               // n
                src1_strides[0] as u32, // stride_a0 (batch stride)
                src1_strides[1] as u32, // stride_a1 (row stride)
                src1_strides[2] as u32, // stride_a2 (column stride)
                src2_strides[0] as u32, // stride_b0 (batch stride)
                src2_strides[1] as u32, // stride_b1 (row stride)
                src2_strides[2] as u32, // stride_b2 (column stride)
                dst_strides[0] as u32,  // stride_c0 (batch stride)
                dst_strides[1] as u32,  // stride_c1 (row stride)
                dst_strides[2] as u32,  // stride_c2 (column stride)
            ];

            // Calculate workgroup dimensions - 8×8×4 threads per workgroup
            let workgroup_size_xy = 8;
            let workgroup_size_z = 4;
            let num_groups_x = (n as u32 + workgroup_size_xy - 1) / workgroup_size_xy;
            let num_groups_y = (m as u32 + workgroup_size_xy - 1) / workgroup_size_xy;
            let num_groups_z = (batch as u32 + workgroup_size_z - 1) / workgroup_size_z;

            Ok((push_constants, num_groups_x, num_groups_y, num_groups_z))
        }

        GPUMemoryOperation::MatMul3D1D => {
            // [batch,m,k] × [k] → [batch,m]
            let batch = src1_dims[0];
            let m = src1_dims[1];
            let k = src1_dims[2];

            let push_constants = vec![
                batch as u32,           // batch
                m as u32,               // m
                k as u32,               // k
                src1_strides[0] as u32, // stride_a0 (batch stride)
                src1_strides[1] as u32, // stride_a1 (row stride)
                src1_strides[2] as u32, // stride_a2 (column stride)
                src2_strides[0] as u32, // stride_b
                dst_strides[0] as u32,  // stride_c0 (batch stride)
                dst_strides[1] as u32,  // stride_c1 (row stride)
            ];

            // Calculate workgroup dimensions - 16×16 threads per workgroup
            let workgroup_size = 16;
            let num_groups_x = (m as u32 + workgroup_size - 1) / workgroup_size;
            let num_groups_y = (batch as u32 + workgroup_size - 1) / workgroup_size;

            Ok((push_constants, num_groups_x, num_groups_y, 1))
        }

        GPUMemoryOperation::MatMul1D3D => {
            // [k] × [batch,k,n] → [batch,n]
            let k = src1_dims[0];
            let batch = src2_dims[0];
            let n = src2_dims[2];

            let push_constants = vec![
                batch as u32,           // batch
                k as u32,               // k
                n as u32,               // n
                src1_strides[0] as u32, // stride_a
                src2_strides[0] as u32, // stride_b0 (batch stride)
                src2_strides[1] as u32, // stride_b1 (row stride)
                src2_strides[2] as u32, // stride_b2 (column stride)
                dst_strides[0] as u32,  // stride_c0 (batch stride)
                dst_strides[1] as u32,  // stride_c1 (column stride)
            ];

            // Calculate workgroup dimensions - 16×16 threads per workgroup
            let workgroup_size = 16;
            let num_groups_x = (n as u32 + workgroup_size - 1) / workgroup_size;
            let num_groups_y = (batch as u32 + workgroup_size - 1) / workgroup_size;

            Ok((push_constants, num_groups_x, num_groups_y, 1))
        }

        _ => Err(format!(
            "Unsupported operation in configure_matmul_operation: {:?}",
            operation
        )
        .into()),
    }
}

fn create_generic_matmul_command_buffer(
    gpu: &GPU,
    command_buffer: vk::CommandBuffer,
    src1_tensor: &ComputeTensor,
    src2_tensor: &ComputeTensor,
    dst_tensor: &ComputeTensor,
) -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        let src1_gpu_mem = match &src1_tensor.data {
            TensorData::GPU { memory, .. } => memory,
            _ => return Err("Source tensor 1 not on GPU".into()),
        };

        let src2_gpu_mem = match &src2_tensor.data {
            TensorData::GPU { memory, .. } => memory,
            _ => return Err("Source tensor 2 not on GPU".into()),
        };

        let dst_gpu_mem = match &dst_tensor.data {
            TensorData::GPU { memory, .. } => memory,
            _ => return Err("Destination tensor not on GPU".into()),
        };

        // Begin command buffer
        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        gpu.get_device()
            .begin_command_buffer(command_buffer, &begin_info)?;

        // Get pipeline
        // Assuming pipeline.descriptor_set_layout and pipeline.pipeline_layout exist
        let pipeline = gpu
            .get_compute_pipelines()
            .get_pipeline(GPUMemoryOperation::MatMul)
            .ok_or("MatMul pipeline not found")?;

        // Prepare UBO data
        let src1_dims = src1_tensor.desc.to_dims();
        let src2_dims = src2_tensor.desc.to_dims();
        let dst_dims = dst_tensor.desc.to_dims();

        let src1_strides = src1_tensor.desc.strides();
        let src2_strides = src2_tensor.desc.strides();
        let dst_strides = dst_tensor.desc.strides();

        const MAX_DIMS: usize = 8; // Must match shader

        let (m, k, n, a_m_axis, a_k_axis, b_k_axis, b_n_axis) =
            analyze_matmul_dimensions(&src1_dims, &src2_dims, &dst_dims)?;

        let mut ubo_data = Vec::<u32>::new();
        // Dimension counts
        ubo_data.push(src1_dims.len() as u32);
        ubo_data.push(src2_dims.len() as u32);
        ubo_data.push(dst_dims.len() as u32);
        // Key dimensions
        ubo_data.push(m as u32);
        ubo_data.push(k as u32);
        ubo_data.push(n as u32);
        // batch_dims (count of batch dimensions in C tensor)
        let batch_dims_count = dst_dims.len().saturating_sub(2);
        ubo_data.push(batch_dims_count as u32);
        // Contraction axis information (matching shader struct order)
        ubo_data.push(a_k_axis as u32); // a_k_axis
        ubo_data.push(b_k_axis as u32); // b_k_axis
        ubo_data.push(a_m_axis as u32); // a_m_axis
        ubo_data.push(b_n_axis as u32); // b_n_axis

        let pad_dims_or_strides = |values: &[usize]| -> [u32; MAX_DIMS] {
            let mut padded = [0u32; MAX_DIMS];
            for (i, &val) in values.iter().enumerate().take(MAX_DIMS) {
                padded[i] = val as u32;
            }
            padded
        };
        // Tensor shapes
        ubo_data.extend_from_slice(&pad_dims_or_strides(&src1_dims));
        ubo_data.extend_from_slice(&pad_dims_or_strides(&src2_dims));
        ubo_data.extend_from_slice(&pad_dims_or_strides(&dst_dims));
        // Tensor strides
        ubo_data.extend_from_slice(&pad_dims_or_strides(&src1_strides));
        ubo_data.extend_from_slice(&pad_dims_or_strides(&src2_strides));
        ubo_data.extend_from_slice(&pad_dims_or_strides(&dst_strides));

        let ubo_data_bytes: Vec<u8> = ubo_data.iter().flat_map(|val| val.to_ne_bytes()).collect();

        let ubo_buffer_info = vk::BufferCreateInfo {
            size: ubo_data_bytes.len() as vk::DeviceSize,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let ubo_buffer = gpu.get_device().create_buffer(&ubo_buffer_info, None)?;

        let ubo_mem_req = gpu.get_device().get_buffer_memory_requirements(ubo_buffer);
        let ubo_memory_type_index = gpu.find_memory_type(
            ubo_mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let ubo_alloc_info = vk::MemoryAllocateInfo {
            allocation_size: ubo_mem_req.size,
            memory_type_index: ubo_memory_type_index,
            ..Default::default()
        };
        let ubo_memory = gpu.get_device().allocate_memory(&ubo_alloc_info, None)?;
        gpu.get_device()
            .bind_buffer_memory(ubo_buffer, ubo_memory, 0)?;

        let ubo_ptr = gpu.get_device().map_memory(
            ubo_memory,
            0,
            ubo_mem_req.size,
            vk::MemoryMapFlags::empty(),
        )?;
        std::ptr::copy_nonoverlapping(
            ubo_data_bytes.as_ptr(),
            ubo_ptr as *mut u8,
            ubo_data_bytes.len(),
        );
        gpu.get_device().unmap_memory(ubo_memory);

        // Descriptor sets
        let set_layouts = [*gpu.get_descriptor_set_layout()];
        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: *gpu.get_descriptor_pool(),
            descriptor_set_count: 1,
            set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };
        let descriptor_set = gpu.get_device().allocate_descriptor_sets(&alloc_info)?[0];

        let src1_buffer_info = vk::DescriptorBufferInfo {
            buffer: src1_gpu_mem.buffer,
            offset: 0,
            range: src1_gpu_mem.size,
        };
        let src2_buffer_info = vk::DescriptorBufferInfo {
            buffer: src2_gpu_mem.buffer,
            offset: 0,
            range: src2_gpu_mem.size,
        };
        let dst_buffer_info = vk::DescriptorBufferInfo {
            buffer: dst_gpu_mem.buffer,
            offset: 0,
            range: dst_gpu_mem.size,
        };
        let ubo_descriptor_buffer_info = vk::DescriptorBufferInfo {
            buffer: ubo_buffer,
            offset: 0,
            range: ubo_data_bytes.len() as vk::DeviceSize,
        };

        let write_descriptor_sets = [
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                buffer_info: &src1_buffer_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                buffer_info: &src2_buffer_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                buffer_info: &dst_buffer_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 3, // UBO binding
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                buffer_info: &ubo_descriptor_buffer_info,
                ..Default::default()
            },
        ];
        gpu.get_device()
            .update_descriptor_sets(&write_descriptor_sets, &[] as &[vk::CopyDescriptorSet]);

        gpu.get_device().cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline,
        );
        gpu.get_device().cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            gpu.get_compute_pipelines().get_layout(),
            0,
            &[descriptor_set],
            &[],
        );

        // Calculate batch size for dispatch (product of C's batch dimensions)
        let mut batch_size_val = 1usize;
        if dst_dims.len() > 2 {
            for i in 0..(dst_dims.len() - 2) {
                batch_size_val *= dst_dims[i];
            }
        }

        let workgroup_size_x = 16; // from shader
        let workgroup_size_y = 16; // from shader
        let workgroup_size_z = 1; // from shader layout(local_size_z = 1)

        let num_groups_x = (n as u32 + workgroup_size_x - 1) / workgroup_size_x;
        let num_groups_y = (m as u32 + workgroup_size_y - 1) / workgroup_size_y;
        let num_groups_z = (batch_size_val as u32 + workgroup_size_z - 1) / workgroup_size_z; // if workgroup_size_z is 1, this is just batch_size_val

        gpu.get_device()
            .cmd_dispatch(command_buffer, num_groups_x, num_groups_y, num_groups_z);

        gpu.get_device().end_command_buffer(command_buffer)?;

        // IMPORTANT: The ubo_buffer and ubo_memory allocated here need to be freed
        // after the command buffer has finished execution. This should be managed
        // by your GPU resource system (e.g., add to a cleanup queue associated
        // with the command buffer's fence).
        // Example: gpu.defer_cleanup(ubo_buffer, ubo_memory);

        Ok(())
    }
}

fn analyze_matmul_dimensions(
    src1_dims: &[usize],
    src2_dims: &[usize],
    dst_dims: &[usize],
) -> Result<(usize, usize, usize, usize, usize, usize, usize), Box<dyn std::error::Error>> {
    if src1_dims.is_empty() || src2_dims.is_empty() {
        return Err("Empty tensor dimensions".into());
    }

    // Find matrix multiplication dimensions based on common patterns

    // Pattern 1: Standard matrix multiplication [m,k] × [k,n] → [m,n]
    if src1_dims.len() == 2 && src2_dims.len() == 2 && dst_dims.len() == 2 {
        let m = src1_dims[0];
        let k1 = src1_dims[1];
        let k2 = src2_dims[0];
        let n = src2_dims[1];

        if k1 == k2 && dst_dims[0] == m && dst_dims[1] == n {
            return Ok((m, k1, n, 0, 1, 0, 1));
        }
    }

    // Pattern 2: Batched matrix multiplication [batch,m,k] × [batch,k,n] → [batch,m,n]
    if src1_dims.len() == 3 && src2_dims.len() == 3 && dst_dims.len() == 3 {
        let batch1 = src1_dims[0];
        let m = src1_dims[1];
        let k1 = src1_dims[2];

        let batch2 = src2_dims[0];
        let k2 = src2_dims[1];
        let n = src2_dims[2];

        if batch1 == batch2
            && k1 == k2
            && dst_dims[0] == batch1
            && dst_dims[1] == m
            && dst_dims[2] == n
        {
            return Ok((m, k1, n, 1, 2, 1, 2));
        }
    }

    // Pattern 3: Higher-dimensional tensor contraction - general case
    // This is a complex analysis that would try to find matching dimensions
    // For now, let's implement a simplified version for common cases

    // Find the innermost dimensions, which are likely the matrix multiply dimensions
    let a_k_axis = src1_dims.len() - 1;
    let b_k_axis = src2_dims.len() - 2;
    let a_m_axis = src1_dims.len() - 2;
    let b_n_axis = src2_dims.len() - 1;

    let k1 = src1_dims[a_k_axis];
    let k2 = src2_dims[b_k_axis];

    if k1 != k2 {
        return Err(format!(
            "Inner dimensions for matrix multiplication don't match: {} vs {}",
            k1, k2
        )
        .into());
    }

    let m = src1_dims[a_m_axis];
    let n = src2_dims[b_n_axis];

    // Check that output shape is compatible
    if dst_dims.len() < 2 || dst_dims[dst_dims.len() - 2] != m || dst_dims[dst_dims.len() - 1] != n
    {
        return Err(format!(
            "Output shape {:?} doesn't match expected dimensions m={}, n={}",
            dst_dims, m, n
        )
        .into());
    }

    // For now, we'll assume a standard matmul pattern, but we can expand this
    // to handle more complex contractions if needed
    Ok((m, k1, n, a_m_axis, a_k_axis, b_k_axis, b_n_axis))
}

fn create_specialized_matmul_command_buffer(
    gpu: &GPU,
    command_buffer: vk::CommandBuffer,
    src1_tensor: &ComputeTensor,
    src2_tensor: &ComputeTensor,
    dst_tensor: &ComputeTensor,
    operation: GPUMemoryOperation,
) -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        let src1_mem = match &src1_tensor.data {
            TensorData::GPU { memory, .. } => memory,
            _ => return Err("Source tensor 1 not in GPU memory".into()),
        };

        let src2_mem = match &src2_tensor.data {
            TensorData::GPU { memory, .. } => memory,
            _ => return Err("Source tensor 2 not in GPU memory".into()),
        };

        let dst_mem = match &dst_tensor.data {
            TensorData::GPU { memory, .. } => memory,
            _ => return Err("Destination tensor not in GPU memory".into()),
        };

        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            next: std::ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            inheritance_info: std::ptr::null(),
        };
        gpu.get_device()
            .begin_command_buffer(command_buffer, &begin_info)?;

        let set_layouts = [*gpu.get_descriptor_set_layout()];
        let alloc_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            next: std::ptr::null(),
            descriptor_pool: *gpu.get_descriptor_pool(),
            descriptor_set_count: 1,
            set_layouts: set_layouts.as_ptr(),
        };

        let descriptor_set = gpu.get_device().allocate_descriptor_sets(&alloc_info)?[0];

        let buffer_infos = [
            vk::DescriptorBufferInfo {
                buffer: src1_mem.buffer,
                offset: 0,
                range: src1_mem.size,
            },
            vk::DescriptorBufferInfo {
                buffer: src2_mem.buffer,
                offset: 0,
                range: src2_mem.size,
            },
            vk::DescriptorBufferInfo {
                buffer: dst_mem.buffer,
                offset: 0,
                range: dst_mem.size,
            },
        ];

        let write_descriptor_sets = [
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                next: std::ptr::null(),
                dst_set: descriptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                buffer_info: &buffer_infos[0],
                image_info: std::ptr::null(),
                texel_buffer_view: std::ptr::null(),
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                next: std::ptr::null(),
                dst_set: descriptor_set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                buffer_info: &buffer_infos[1],
                image_info: std::ptr::null(),
                texel_buffer_view: std::ptr::null(),
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                next: std::ptr::null(),
                dst_set: descriptor_set,
                dst_binding: 2,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                buffer_info: &buffer_infos[2],
                image_info: std::ptr::null(),
                texel_buffer_view: std::ptr::null(),
            },
        ];

        gpu.get_device()
            .update_descriptor_sets(&write_descriptor_sets, &[] as &[vk::CopyDescriptorSet]);

        let pipeline = gpu
            .get_compute_pipelines()
            .get_pipeline(operation)
            .ok_or(format!("{:?} pipeline not found", operation))?;

        gpu.get_device().cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline,
        );

        gpu.get_device().cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            gpu.get_compute_pipelines().get_layout(),
            0,
            &[descriptor_set],
            &[],
        );

        // Configure operation-specific parameters and dispatch dimensions
        let (push_constants, dispatch_x, dispatch_y, dispatch_z) =
            configure_matmul_operation(operation, src1_tensor, src2_tensor, dst_tensor)?;

        // Push constants to the shader
        gpu.get_device().cmd_push_constants(
            command_buffer,
            gpu.get_compute_pipelines().get_layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            std::slice::from_raw_parts(
                push_constants.as_ptr() as *const u8,
                push_constants.len() * std::mem::size_of::<u32>(),
            ),
        );

        gpu.get_device()
            .cmd_dispatch(command_buffer, dispatch_x, dispatch_y, dispatch_z);

        gpu.get_device().end_command_buffer(command_buffer)?;

        Ok(())
    }
}
