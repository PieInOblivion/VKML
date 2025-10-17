use crate::instruction::matmul::f32_f32_f32_cpu::f32_f32_f32_cpu;
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

    fn record_into_command_buffer(
        &self,
        gpu: &Gpu,
        command_buffer: vk::CommandBuffer,
        cm: &ComputeManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let src1_tensor = cm.tensor_read(self.src1);
        let src2_tensor = cm.tensor_read(self.src2);
        let dst_tensor = cm.tensor_read(self.dst);

        // Validate datatypes first and determine operation variant
        let src1_dims = src1_tensor.desc.dims();
        let src2_dims = src2_tensor.desc.dims();
        let src1_dtype = src1_tensor.desc.data_type();
        let src2_dtype = src2_tensor.desc.data_type();
        let dst_dtype = dst_tensor.desc.data_type();

        // Only support Float triplet on GPU for now
        match (src1_dtype, src2_dtype, dst_dtype) {
            (DataType::Float, DataType::Float, DataType::Float) => {
                let operation = determine_matmul_variant(src1_dims, src2_dims);

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
            _ => Err(format!(
                "GPU MatMul unimplemented for DataType src1:{:?}, src2:{:?}, dst:{:?}",
                src1_dtype, src2_dtype, dst_dtype
            )
            .into()),
        }
    }

    fn clone_box(&self) -> Box<dyn Instruction> {
        Box::new(self.clone())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        let src1_tensor = cm.tensor_read(self.src1);
        let src2_tensor = cm.tensor_read(self.src2);
        let dst_tensor = cm.tensor_write(self.dst);

        let src1_dtype = src1_tensor.desc.data_type();
        let src2_dtype = src2_tensor.desc.data_type();
        let dst_dtype = dst_tensor.desc.data_type();

        // dims are i64 internally; convert to usize for CPU indexing/math
        let src1_dims_i64 = src1_tensor.desc.dims();
        let src2_dims_i64 = src2_tensor.desc.dims();
        let dst_dims_i64 = dst_tensor.desc.dims();
        let src1_dims: Vec<usize> = src1_dims_i64.iter().map(|&d| d as usize).collect();
        let src2_dims: Vec<usize> = src2_dims_i64.iter().map(|&d| d as usize).collect();
        let dst_dims: Vec<usize> = dst_dims_i64.iter().map(|&d| d as usize).collect();

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
                src1_dtype, src2_dtype, dst_dtype
            ),
        }
    }
}

fn determine_matmul_variant(src1_dims: &[i64], src2_dims: &[i64]) -> GPUOperation {
    let a_rank = src1_dims.len();
    let b_rank = src2_dims.len();

    if a_rank == 0 || b_rank == 0 {
        panic!(
            "MatMul: zero-rank tensor not supported (a_rank={}, b_rank={})",
            a_rank, b_rank
        );
    }

    // Prefer specialised kernels for common small-rank combinations for performance.
    match (a_rank, b_rank) {
        (1, 2) => GPUOperation::MatMul1D2D_F32_F32_F32,
        (2, 1) => GPUOperation::MatMul2D1D_F32_F32_F32,
        (2, 2) => GPUOperation::MatMul2D2D_F32_F32_F32,
        (2, 3) => GPUOperation::MatMul2D3D_F32_F32_F32,
        (3, 2) => GPUOperation::MatMul3D2D_F32_F32_F32,
        (3, 3) => GPUOperation::MatMul3D3D_F32_F32_F32,
        (3, 1) => GPUOperation::MatMul3D1D_F32_F32_F32,
        (1, 3) => GPUOperation::MatMul1D3D_F32_F32_F32,
        _ => unimplemented!(
            "Unsupported MatMul Dimensions: a_rank:{}, b_rank{}",
            a_rank,
            b_rank
        ),
    }
}

fn configure_matmul_operation(
    operation: GPUOperation,
    src1_tensor: &Tensor,
    src2_tensor: &Tensor,
    dst_tensor: &Tensor,
) -> Result<(Vec<u32>, u32, u32, u32), Box<dyn std::error::Error>> {
    let src1_dims = src1_tensor.desc.dims();
    let src2_dims = src2_tensor.desc.dims();

    let src1_strides = src1_tensor.desc.strides();
    let src2_strides = src2_tensor.desc.strides();
    let dst_strides = dst_tensor.desc.strides();

    match operation {
        GPUOperation::MatMul1D2D_F32_F32_F32 => {
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
            let num_groups_x = (n as u32).div_ceil(workgroup_size);

            Ok((push_constants, num_groups_x, 1, 1))
        }

        GPUOperation::MatMul2D1D_F32_F32_F32 => {
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
            let num_groups_x = (m as u32).div_ceil(workgroup_size);

            Ok((push_constants, num_groups_x, 1, 1))
        }

        GPUOperation::MatMul2D2D_F32_F32_F32 => {
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
            let num_groups_x = (n as u32).div_ceil(workgroup_size);
            let num_groups_y = (m as u32).div_ceil(workgroup_size);

            Ok((push_constants, num_groups_x, num_groups_y, 1))
        }

        GPUOperation::MatMul2D3D_F32_F32_F32 => {
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
            let num_groups_x = (n as u32).div_ceil(workgroup_size_xy);
            let num_groups_y = (m as u32).div_ceil(workgroup_size_xy);
            let num_groups_z = (batch as u32).div_ceil(workgroup_size_z);

            Ok((push_constants, num_groups_x, num_groups_y, num_groups_z))
        }

        GPUOperation::MatMul3D2D_F32_F32_F32 => {
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
            let num_groups_x = (n as u32).div_ceil(workgroup_size_xy);
            let num_groups_y = (m as u32).div_ceil(workgroup_size_xy);
            let num_groups_z = (batch as u32).div_ceil(workgroup_size_z);

            Ok((push_constants, num_groups_x, num_groups_y, num_groups_z))
        }

        GPUOperation::MatMul3D3D_F32_F32_F32 => {
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
            let num_groups_x = (n as u32).div_ceil(workgroup_size_xy);
            let num_groups_y = (m as u32).div_ceil(workgroup_size_xy);
            let num_groups_z = (batch as u32).div_ceil(workgroup_size_z);

            Ok((push_constants, num_groups_x, num_groups_y, num_groups_z))
        }

        GPUOperation::MatMul3D1D_F32_F32_F32 => {
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
            let num_groups_x = (m as u32).div_ceil(workgroup_size);
            let num_groups_y = (batch as u32).div_ceil(workgroup_size);

            Ok((push_constants, num_groups_x, num_groups_y, 1))
        }

        GPUOperation::MatMul1D3D_F32_F32_F32 => {
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
            let num_groups_x = (n as u32).div_ceil(workgroup_size);
            let num_groups_y = (batch as u32).div_ceil(workgroup_size);

            Ok((push_constants, num_groups_x, num_groups_y, 1))
        }

        _ => Err(format!(
            "Unsupported operation in configure_matmul_operation: {:?}",
            operation
        )
        .into()),
    }
}

fn create_specialized_matmul_command_buffer(
    gpu: &Gpu,
    command_buffer: vk::CommandBuffer,
    src1_tensor: &Tensor,
    src2_tensor: &Tensor,
    dst_tensor: &Tensor,
    operation: GPUOperation,
) -> Result<(), Box<dyn std::error::Error>> {
    let src1_mem = src1_tensor.get_gpu_memory_or_panic();
    let src2_mem = src2_tensor.get_gpu_memory_or_panic();
    let dst_mem = dst_tensor.get_gpu_memory_or_panic();

    // Choose local workgroup size appropriate for the specialised kernel later and bind pipeline
    // We'll pick based on the operation's expected tile sizes
    let local_size = match operation {
        GPUOperation::MatMul1D2D_F32_F32_F32
        | GPUOperation::MatMul2D1D_F32_F32_F32
        | GPUOperation::MatMul3D1D_F32_F32_F32
        | GPUOperation::MatMul1D3D_F32_F32_F32 => gpu.optimal_workgroup_size_1d(1), // will be overridden per-dispatch below for total work
        GPUOperation::MatMul2D2D_F32_F32_F32
        | GPUOperation::MatMul3D2D_F32_F32_F32
        | GPUOperation::MatMul2D3D_F32_F32_F32
        | GPUOperation::MatMul3D3D_F32_F32_F32 => gpu.optimal_workgroup_size_2d(16, 16),
        _ => gpu.optimal_workgroup_size_1d(1),
    };

    gpu.bind_compute_pipeline(command_buffer, operation, local_size);
    gpu.bind_storage_buffers(command_buffer, &[src1_mem, src2_mem, dst_mem]);

    // Configure operation-specific parameters and dispatch dimensions
    let (push_constants, dispatch_x, dispatch_y, dispatch_z) =
        configure_matmul_operation(operation, src1_tensor, src2_tensor, dst_tensor)?;

    // Serialize push-constants (Vec<u32> -> &[u8])
    let mut pc_bytes_vec: Vec<u8> =
        Vec::with_capacity(push_constants.len() * std::mem::size_of::<u32>());
    for v in &push_constants {
        pc_bytes_vec.extend_from_slice(&v.to_ne_bytes());
    }

    // Push constants
    gpu.bind_push_constants(command_buffer, pc_bytes_vec.as_slice());

    // Convert dispatch counts from (groups_x, groups_y, groups_z) into total work extents
    // depending on the kernel we select local_size to match shader tiles and then
    // provide total work sizes so Gpu::dispatch computes the number of workgroups.
    let work_size: [u64; 3] = match operation {
        GPUOperation::MatMul1D2D_F32_F32_F32 | GPUOperation::MatMul2D1D_F32_F32_F32 => {
            // 1D kernels: dispatch along single dimension (n or m)
            [dispatch_x as u64, 1u64, 1u64]
        }
        GPUOperation::MatMul2D2D_F32_F32_F32 => {
            // 2D kernel: provide (n, m, batch)
            let n_u = dispatch_x as u64 * 16u64; // approximate total cols = groups_x * tile_x
            let m_u = dispatch_y as u64 * 16u64; // approximate total rows = groups_y * tile_y
            let batch_u = dispatch_z as u64; // batch count
            [n_u, m_u, batch_u]
        }
        GPUOperation::MatMul2D3D_F32_F32_F32
        | GPUOperation::MatMul3D2D_F32_F32_F32
        | GPUOperation::MatMul3D3D_F32_F32_F32 => {
            // 3D kernels: groups_x/groups_y correspond to tiles in n/m, groups_z to batch
            let n_u = dispatch_x as u64 * 8u64; // shader tiles are 8 in xy for these kernels
            let m_u = dispatch_y as u64 * 8u64;
            let batch_u = dispatch_z as u64 * 4u64; // z tile size 4
            [n_u, m_u, batch_u]
        }
        GPUOperation::MatMul3D1D_F32_F32_F32 | GPUOperation::MatMul1D3D_F32_F32_F32 => {
            [dispatch_x as u64, dispatch_y as u64, dispatch_z as u64]
        }
        _ => [dispatch_x as u64, dispatch_y as u64, dispatch_z as u64],
    };

    gpu.dispatch(command_buffer, local_size, work_size);

    Ok(())
}
