use crate::error::VKMLError;
use crate::instruction::matmul::f32_f32_f32_cpu::f32_f32_f32_cpu;
use crate::instruction::matmul::push_constants::{
    MatMul1D2DPushConstants, MatMul1D3DPushConstants, MatMul2D1DPushConstants,
    MatMul2D2DPushConstants, MatMul2D3DPushConstants, MatMul3D1DPushConstants,
    MatMul3D2DPushConstants, MatMul3D3DPushConstants,
};
use crate::utils::bytes::as_bytes;
use crate::{
    ComputeManager,
    gpu::vk_gpu::Gpu,
    instruction::{gpu_operations::GPUOperation, instruction::Instruction},
    tensor::Tensor,
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

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

    fn record_into_command_buffer(
        &self,
        gpu: &Gpu,
        command_buffer: vk::CommandBuffer,
        cm: &ComputeManager,
    ) -> Result<(), VKMLError> {
        let src1_tensor = cm.tensor_read(self.src1);
        let src2_tensor = cm.tensor_read(self.src2);
        let dst_tensor = cm.tensor_read(self.dst);

        let src1_dtype = src1_tensor.desc.data_type();
        let src2_dtype = src2_tensor.desc.data_type();
        let dst_dtype = dst_tensor.desc.data_type();

        // Determine GPU operation based on dimensions and datatypes
        let operation = determine_operation(
            src1_tensor.desc.dims(),
            src2_tensor.desc.dims(),
            src1_dtype,
            src2_dtype,
            dst_dtype,
        )?;

        execute_gpu_matmul(
            gpu,
            command_buffer,
            src1_tensor,
            src2_tensor,
            dst_tensor,
            operation,
        )
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        let src1_tensor = cm.tensor_read(self.src1);
        let src2_tensor = cm.tensor_read(self.src2);
        let dst_tensor = cm.tensor_write(self.dst);

        let src1_dtype = src1_tensor.desc.data_type();
        let src2_dtype = src2_tensor.desc.data_type();
        let dst_dtype = dst_tensor.desc.data_type();

        let src1_dims: Vec<usize> = src1_tensor
            .desc
            .dims()
            .iter()
            .map(|&d| d as usize)
            .collect();
        let src2_dims: Vec<usize> = src2_tensor
            .desc
            .dims()
            .iter()
            .map(|&d| d as usize)
            .collect();
        let dst_dims: Vec<usize> = dst_tensor.desc.dims().iter().map(|&d| d as usize).collect();

        let src1_bytes = src1_tensor.get_cpu_memory_slice_or_panic();
        let src2_bytes = src2_tensor.get_cpu_memory_slice_or_panic();
        let dst_bytes = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match (src1_dtype, src2_dtype, dst_dtype) {
            (DataType::Float, DataType::Float, DataType::Float) => {
                f32_f32_f32_cpu(
                    src1_dims, src2_dims, dst_dims, src1_bytes, src2_bytes, dst_bytes,
                );
            }
            _ => unimplemented!(
                "CPU MatMul: unimplemented for DataType src1:{:?}, src2:{:?}, dst:{:?}",
                src1_dtype,
                src2_dtype,
                dst_dtype
            ),
        }
    }
}

/// Determine which GPU operation to use based on tensor dimensions and datatypes
fn determine_operation(
    src1_dims: &[i64],
    src2_dims: &[i64],
    src1_dtype: DataType,
    src2_dtype: DataType,
    dst_dtype: DataType,
) -> Result<GPUOperation, VKMLError> {
    let a_rank = src1_dims.len();
    let b_rank = src2_dims.len();

    if a_rank == 0 || b_rank == 0 {
        return Err(VKMLError::Instruction(format!(
            "MatMul: zero-rank tensor not supported (a_rank={}, b_rank={})",
            a_rank, b_rank
        )));
    }

    // Map (shape, datatypes) to GPUOperation
    match (src1_dtype, src2_dtype, dst_dtype) {
        (DataType::Float, DataType::Float, DataType::Float) => match (a_rank, b_rank) {
            (1, 2) => Ok(GPUOperation::MatMul1D2D_F32_F32_F32),
            (2, 1) => Ok(GPUOperation::MatMul2D1D_F32_F32_F32),
            (2, 2) => Ok(GPUOperation::MatMul2D2D_F32_F32_F32),
            (2, 3) => Ok(GPUOperation::MatMul2D3D_F32_F32_F32),
            (3, 2) => Ok(GPUOperation::MatMul3D2D_F32_F32_F32),
            (3, 3) => Ok(GPUOperation::MatMul3D3D_F32_F32_F32),
            (3, 1) => Ok(GPUOperation::MatMul3D1D_F32_F32_F32),
            (1, 3) => Ok(GPUOperation::MatMul1D3D_F32_F32_F32),
            _ => Err(VKMLError::Instruction(format!(
                "Unsupported MatMul dimensions: a_rank:{}, b_rank:{}",
                a_rank, b_rank
            ))),
        },
        (DataType::Float16, DataType::Float16, DataType::Float16) => match (a_rank, b_rank) {
            (2, 2) => Ok(GPUOperation::MatMul2D2D_F16_F16_F16), // Will be replaced by coop matrix if available
            _ => Err(VKMLError::Instruction(format!(
                "Unsupported F16 MatMul dimensions: a_rank:{}, b_rank:{}",
                a_rank, b_rank
            ))),
        },
        _ => Err(VKMLError::Instruction(format!(
            "GPU MatMul unimplemented for DataType src1:{:?}, src2:{:?}, dst:{:?}",
            src1_dtype, src2_dtype, dst_dtype
        ))),
    }
}

/// Execute GPU MatMul operation using specialised shaders
fn execute_gpu_matmul(
    gpu: &Gpu,
    command_buffer: vk::CommandBuffer,
    src1_tensor: &Tensor,
    src2_tensor: &Tensor,
    dst_tensor: &Tensor,
    operation: GPUOperation,
) -> Result<(), VKMLError> {
    let src1_mem = src1_tensor.get_gpu_memory_or_panic();
    let src2_mem = src2_tensor.get_gpu_memory_or_panic();
    let dst_mem = dst_tensor.get_gpu_memory_or_panic();

    let src1_dims = src1_tensor.desc.dims();
    let src2_dims = src2_tensor.desc.dims();
    let src1_strides = src1_tensor.desc.strides();
    let src2_strides = src2_tensor.desc.strides();
    let dst_strides = dst_tensor.desc.strides();

    // Configure based on operation type
    // Pass actual output dimensions to optimal_workgroup_size_* and dispatch
    let (local_size, push_constants_bytes, work_size) = match operation {
        GPUOperation::MatMul1D2D_F32_F32_F32 => {
            // [k] × [k,n] → [n]
            let k = src1_dims[0];
            let n = src2_dims[1];

            let pc = MatMul1D2DPushConstants {
                k: k as u32,
                n: n as u32,
                stride_a: src1_strides[0] as u32,
                stride_b0: src2_strides[0] as u32,
                stride_b1: src2_strides[1] as u32,
                stride_c: dst_strides[0] as u32,
            };

            (
                gpu.optimal_workgroup_size_1d(n as u64),
                as_bytes(&pc).to_vec(),
                [n as u64, 1, 1],
            )
        }

        GPUOperation::MatMul2D1D_F32_F32_F32 => {
            // [m,k] × [k] → [m]
            let m = src1_dims[0];
            let k = src1_dims[1];

            let pc = MatMul2D1DPushConstants {
                m: m as u32,
                k: k as u32,
                stride_a0: src1_strides[0] as u32,
                stride_a1: src1_strides[1] as u32,
                stride_b: src2_strides[0] as u32,
                stride_c: dst_strides[0] as u32,
            };

            (
                gpu.optimal_workgroup_size_1d(m as u64),
                as_bytes(&pc).to_vec(),
                [m as u64, 1, 1],
            )
        }

        GPUOperation::MatMul2D2D_F32_F32_F32 | GPUOperation::MatMul2D2D_F16_F16_F16 => {
            // [m,k] × [k,n] → [m,n]
            let m = src1_dims[0];
            let k = src1_dims[1];
            let n = src2_dims[1];

            let pc = MatMul2D2DPushConstants {
                m: m as u32,
                k: k as u32,
                n: n as u32,
                stride_a0: src1_strides[0] as u32,
                stride_a1: src1_strides[1] as u32,
                stride_b0: src2_strides[0] as u32,
                stride_b1: src2_strides[1] as u32,
                stride_c0: dst_strides[0] as u32,
                stride_c1: dst_strides[1] as u32,
            };

            (
                gpu.optimal_workgroup_size_2d(m as u64, n as u64),
                as_bytes(&pc).to_vec(),
                [n as u64, m as u64, 1],
            )
        }

        GPUOperation::MatMul2D3D_F32_F32_F32 => {
            // [m,k] × [batch,k,n] → [batch,m,n]
            let m = src1_dims[0];
            let k = src1_dims[1];
            let batch = src2_dims[0];
            let n = src2_dims[2];

            let pc = MatMul2D3DPushConstants {
                batch: batch as u32,
                m: m as u32,
                k: k as u32,
                n: n as u32,
                stride_a0: src1_strides[0] as u32,
                stride_a1: src1_strides[1] as u32,
                stride_b0: src2_strides[0] as u32,
                stride_b1: src2_strides[1] as u32,
                stride_b2: src2_strides[2] as u32,
                stride_c0: dst_strides[0] as u32,
                stride_c1: dst_strides[1] as u32,
                stride_c2: dst_strides[2] as u32,
            };

            (
                gpu.optimal_workgroup_size_3d(n as u64, m as u64, batch as u64),
                as_bytes(&pc).to_vec(),
                [n as u64, m as u64, batch as u64],
            )
        }

        GPUOperation::MatMul3D2D_F32_F32_F32 => {
            // [batch,m,k] × [k,n] → [batch,m,n]
            let batch = src1_dims[0];
            let m = src1_dims[1];
            let k = src1_dims[2];
            let n = src2_dims[1];

            let pc = MatMul3D2DPushConstants {
                batch: batch as u32,
                m: m as u32,
                k: k as u32,
                n: n as u32,
                stride_a0: src1_strides[0] as u32,
                stride_a1: src1_strides[1] as u32,
                stride_a2: src1_strides[2] as u32,
                stride_b0: src2_strides[0] as u32,
                stride_b1: src2_strides[1] as u32,
                stride_c0: dst_strides[0] as u32,
                stride_c1: dst_strides[1] as u32,
                stride_c2: dst_strides[2] as u32,
            };

            (
                gpu.optimal_workgroup_size_3d(n as u64, m as u64, batch as u64),
                as_bytes(&pc).to_vec(),
                [n as u64, m as u64, batch as u64],
            )
        }

        GPUOperation::MatMul3D3D_F32_F32_F32 => {
            // [batch,m,k] × [batch,k,n] → [batch,m,n]
            let batch = src1_dims[0];
            let m = src1_dims[1];
            let k = src1_dims[2];
            let n = src2_dims[2];

            let pc = MatMul3D3DPushConstants {
                batch: batch as u32,
                m: m as u32,
                k: k as u32,
                n: n as u32,
                stride_a0: src1_strides[0] as u32,
                stride_a1: src1_strides[1] as u32,
                stride_a2: src1_strides[2] as u32,
                stride_b0: src2_strides[0] as u32,
                stride_b1: src2_strides[1] as u32,
                stride_b2: src2_strides[2] as u32,
                stride_c0: dst_strides[0] as u32,
                stride_c1: dst_strides[1] as u32,
                stride_c2: dst_strides[2] as u32,
            };

            (
                gpu.optimal_workgroup_size_3d(n as u64, m as u64, batch as u64),
                as_bytes(&pc).to_vec(),
                [n as u64, m as u64, batch as u64],
            )
        }

        GPUOperation::MatMul3D1D_F32_F32_F32 => {
            // [batch,m,k] × [k] → [batch,m]
            let batch = src1_dims[0];
            let m = src1_dims[1];
            let k = src1_dims[2];

            let pc = MatMul3D1DPushConstants {
                batch: batch as u32,
                m: m as u32,
                k: k as u32,
                stride_a0: src1_strides[0] as u32,
                stride_a1: src1_strides[1] as u32,
                stride_a2: src1_strides[2] as u32,
                stride_b: src2_strides[0] as u32,
                stride_c0: dst_strides[0] as u32,
                stride_c1: dst_strides[1] as u32,
            };

            (
                gpu.optimal_workgroup_size_2d(m as u64, batch as u64),
                as_bytes(&pc).to_vec(),
                [m as u64, batch as u64, 1],
            )
        }

        GPUOperation::MatMul1D3D_F32_F32_F32 => {
            // [k] × [batch,k,n] → [batch,n]
            let k = src1_dims[0];
            let batch = src2_dims[0];
            let n = src2_dims[2];

            let pc = MatMul1D3DPushConstants {
                batch: batch as u32,
                k: k as u32,
                n: n as u32,
                stride_a: src1_strides[0] as u32,
                stride_b0: src2_strides[0] as u32,
                stride_b1: src2_strides[1] as u32,
                stride_b2: src2_strides[2] as u32,
                stride_c0: dst_strides[0] as u32,
                stride_c1: dst_strides[1] as u32,
            };

            (
                gpu.optimal_workgroup_size_2d(n as u64, batch as u64),
                as_bytes(&pc).to_vec(),
                [n as u64, batch as u64, 1],
            )
        }

        _ => {
            return Err(VKMLError::Instruction(format!(
                "Unsupported MatMul operation: {:?}",
                operation
            )));
        }
    };

    // Cooperative matrix optimization for F16 2D×2D MatMul
    if operation == GPUOperation::MatMul2D2D_F16_F16_F16
        && gpu.subgroup_size() == 32
        && let Some(coop_shapes) = gpu.extensions().get_coop_matrix_sizes(
            DataType::Float16,
            DataType::Float16,
            DataType::Float16,
            DataType::Float16,
        )
    {
        // Check if 16x16x16 cooperative matrix is supported
        let has_16x16x16 = coop_shapes
            .iter()
            .any(|shape| shape.m == 16 && shape.n == 16 && shape.k == 16);

        if has_16x16x16 {
            // Extract dimensions from push constants for cooperative matrix dispatch
            let m = src1_dims[0];
            let n = src2_dims[1];

            // For cooperative matrices, workgroup size is hardcoded in shader
            let coop_local_size = [32, 1, 1];
            let num_tiles_x = (n as usize).div_ceil(16); // ceil(n / 16)
            let num_tiles_y = (m as usize).div_ceil(16); // ceil(m / 16)
            let binding_count = 3; // src1, src2, dst

            gpu.bind_compute_pipeline(
                command_buffer,
                GPUOperation::MatMul2D2D_F16_F16_F16_Coop_16_16_16_SG_32,
                coop_local_size,
                binding_count,
            );
            gpu.bind_storage_buffers(command_buffer, &[src1_mem, src2_mem, dst_mem]);
            gpu.bind_push_constants(command_buffer, binding_count, &push_constants_bytes);

            // Dispatch workgroups: one per tile
            gpu.dispatch(
                command_buffer,
                coop_local_size,
                [num_tiles_x as u64, num_tiles_y as u64, 1],
            );

            return Ok(());
        }
    }

    // Standard pipeline path
    let binding_count = 3; // src1, src2, dst
    gpu.bind_compute_pipeline(command_buffer, operation, local_size, binding_count);
    gpu.bind_storage_buffers(command_buffer, &[src1_mem, src2_mem, dst_mem]);

    gpu.bind_push_constants(command_buffer, binding_count, &push_constants_bytes);

    gpu.dispatch(command_buffer, local_size, work_size);

    Ok(())
}
