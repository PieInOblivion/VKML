use crate::instruction::matmul::f32_cpu::f32_cpu;
use crate::{
    ComputeManager,
    gpu::vk_gpu::Gpu,
    instruction::{gpu_operations::GPUOperation, instruction::Instruction},
    tensor::tensor::Tensor,
    tensor_graph::tensor_graph::TensorId,
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

        // Collect dimension slices once for readability
        let src1_dims = src1_tensor.desc.dims();
        let src2_dims = src2_tensor.desc.dims();
        let operation = determine_matmul_variant(src1_dims, src2_dims);

        if operation == GPUOperation::MatMul_F32 {
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

    fn execute_cpu(&self, cm: &ComputeManager) {
        let src1_tensor = cm.tensor_read(self.src1);
        let src2_tensor = cm.tensor_read(self.src2);
        let dst_tensor = cm.tensor_write(self.dst);

        let dtype = dst_tensor.desc.data_type();

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

        match dtype {
            DataType::Float => {
                f32_cpu(
                    src1_dims, src2_dims, dst_dims, src1_bytes, src2_bytes, dst_bytes,
                );
            }
            other => panic!("MatMul: unimplemented CPU for DataType {:?}", other),
        }
    }
}

fn determine_matmul_variant(src1_dims: &[i64], src2_dims: &[i64]) -> GPUOperation {
    const MAX_DIMS: usize = 6;

    let a_rank = src1_dims.len();
    let b_rank = src2_dims.len();

    if a_rank == 0 || b_rank == 0 {
        panic!(
            "MatMul: zero-rank tensor not supported (a_rank={}, b_rank={})",
            a_rank, b_rank
        );
    }

    if a_rank > MAX_DIMS || b_rank > MAX_DIMS {
        panic!(
            "MatMul: tensor rank too large for push-constant generic path (max {}), got a_rank={}, b_rank={}",
            MAX_DIMS, a_rank, b_rank
        );
    }

    // Prefer specialised kernels for common small-rank combinations for performance.
    match (a_rank, b_rank) {
        (1, 2) => GPUOperation::MatMul1D2D_F32,
        (2, 1) => GPUOperation::MatMul2D1D_F32,
        (2, 2) => GPUOperation::MatMul2D2D_F32,
        (2, 3) => GPUOperation::MatMul2D3D_F32,
        (3, 2) => GPUOperation::MatMul3D2D_F32,
        (3, 3) => GPUOperation::MatMul3D3D_F32,
        (3, 1) => GPUOperation::MatMul3D1D_F32,
        (1, 3) => GPUOperation::MatMul1D3D_F32,
        _ => GPUOperation::MatMul_F32,
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
        GPUOperation::MatMul1D2D_F32 => {
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

        GPUOperation::MatMul2D1D_F32 => {
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

        GPUOperation::MatMul2D2D_F32 => {
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

        GPUOperation::MatMul2D3D_F32 => {
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

        GPUOperation::MatMul3D2D_F32 => {
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

        GPUOperation::MatMul3D3D_F32 => {
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

        GPUOperation::MatMul3D1D_F32 => {
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

        GPUOperation::MatMul1D3D_F32 => {
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

fn create_generic_matmul_command_buffer(
    gpu: &Gpu,
    command_buffer: vk::CommandBuffer,
    src1_tensor: &Tensor,
    src2_tensor: &Tensor,
    dst_tensor: &Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    let src1_gpu_mem = src1_tensor.get_gpu_memory_or_panic();
    let src2_gpu_mem = src2_tensor.get_gpu_memory_or_panic();
    let dst_gpu_mem = dst_tensor.get_gpu_memory_or_panic();

    // Prepare push-constant metadata (we limit shapes/strides to MAX_DIMS=6 to fit in 128 bytes)
    let src1_dims = src1_tensor.desc.dims();
    let src2_dims = src2_tensor.desc.dims();
    let dst_dims = dst_tensor.desc.dims();

    let src1_dims_usize: Vec<usize> = src1_dims.iter().map(|&d| d as usize).collect();
    let src2_dims_usize: Vec<usize> = src2_dims.iter().map(|&d| d as usize).collect();
    let dst_dims_usize: Vec<usize> = dst_dims.iter().map(|&d| d as usize).collect();

    let src1_strides = src1_tensor.desc.strides();
    let src2_strides = src2_tensor.desc.strides();
    let dst_strides = dst_tensor.desc.strides();

    let (m, k, n, a_m_axis, a_k_axis, b_k_axis, b_n_axis) =
        analyze_matmul_dimensions(&src1_dims_usize, &src2_dims_usize, &dst_dims_usize)?;

    let pack_pairs = |vals: &[usize]| -> [u32; 3] {
        let mut out = [0u32; 3];
        for i in 0..3 {
            let lo_idx = i * 2;
            let hi_idx = lo_idx + 1;
            let lo = if lo_idx < vals.len() {
                vals[lo_idx] as u32
            } else {
                0u32
            } & 0xFFFFu32;
            let hi = if hi_idx < vals.len() {
                vals[hi_idx] as u32
            } else {
                0u32
            } & 0xFFFFu32;
            out[i] = (hi << 16) | lo;
        }
        out
    };

    let a_shape_packed = pack_pairs(&src1_dims_usize);
    let b_shape_packed = pack_pairs(&src2_dims_usize);
    let c_shape_packed = pack_pairs(&dst_dims_usize);
    let a_strides_packed = pack_pairs(&src1_strides);
    let b_strides_packed = pack_pairs(&src2_strides);
    let c_strides_packed = pack_pairs(&dst_strides);

    // Build push-constant array in the same order as GLSL struct (packed arrays)
    let batch_dims_count = dst_dims_usize.len().saturating_sub(2);
    let mut pc: Vec<u32> = vec![
        src1_dims_usize.len() as u32,
        src2_dims_usize.len() as u32,
        dst_dims_usize.len() as u32,
        m as u32,
        k as u32,
        n as u32,
        batch_dims_count as u32,
        a_k_axis as u32,
        b_k_axis as u32,
        a_m_axis as u32,
        b_n_axis as u32,
    ];

    pc.extend_from_slice(&a_shape_packed);
    pc.extend_from_slice(&b_shape_packed);
    pc.extend_from_slice(&c_shape_packed);
    pc.extend_from_slice(&a_strides_packed);
    pc.extend_from_slice(&b_strides_packed);
    pc.extend_from_slice(&c_strides_packed);

    let local_size = gpu.optimal_workgroup_size_2d(n as u64, m as u64);
    gpu.bind_compute_pipeline(command_buffer, GPUOperation::MatMul_F32, local_size);
    gpu.bind_storage_buffers(command_buffer, &[src1_gpu_mem, src2_gpu_mem, dst_gpu_mem]);

    // push constants (u32 array as bytes in native-endian)
    let mut pc_bytes: Vec<u8> = Vec::with_capacity(pc.len() * 4);
    for v in &pc {
        pc_bytes.extend_from_slice(&v.to_ne_bytes());
    }
    gpu.bind_push_constants(command_buffer, pc_bytes.as_slice());

    // Calculate batch size for dispatch (product of C's batch dimensions)
    let mut batch_size_val = 1usize;
    if dst_dims_usize.len() > 2 {
        for i in 0..(dst_dims_usize.len() - 2) {
            batch_size_val *= dst_dims_usize[i];
        }
    }

    // Dispatch using total work extents; Gpu::dispatch will ceil-divide by local_size
    gpu.dispatch(
        command_buffer,
        local_size,
        [n as u64, m as u64, batch_size_val as u64],
    );

    Ok(())
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
        GPUOperation::MatMul1D2D_F32
        | GPUOperation::MatMul2D1D_F32
        | GPUOperation::MatMul3D1D_F32
        | GPUOperation::MatMul1D3D_F32 => gpu.optimal_workgroup_size_1d(1), // will be overridden per-dispatch below for total work
        GPUOperation::MatMul2D2D_F32
        | GPUOperation::MatMul3D2D_F32
        | GPUOperation::MatMul2D3D_F32
        | GPUOperation::MatMul3D3D_F32 => gpu.optimal_workgroup_size_2d(16, 16),
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
        GPUOperation::MatMul1D2D_F32 | GPUOperation::MatMul2D1D_F32 => {
            // 1D kernels: dispatch along single dimension (n or m)
            [dispatch_x as u64, 1u64, 1u64]
        }
        GPUOperation::MatMul2D2D_F32 => {
            // 2D kernel: provide (n, m, batch)
            let n_u = dispatch_x as u64 * 16u64; // approximate total cols = groups_x * tile_x
            let m_u = dispatch_y as u64 * 16u64; // approximate total rows = groups_y * tile_y
            let batch_u = dispatch_z as u64; // batch count
            [n_u, m_u, batch_u]
        }
        GPUOperation::MatMul2D3D_F32
        | GPUOperation::MatMul3D2D_F32
        | GPUOperation::MatMul3D3D_F32 => {
            // 3D kernels: groups_x/groups_y correspond to tiles in n/m, groups_z to batch
            let n_u = dispatch_x as u64 * 8u64; // shader tiles are 8 in xy for these kernels
            let m_u = dispatch_y as u64 * 8u64;
            let batch_u = dispatch_z as u64 * 4u64; // z tile size 4
            [n_u, m_u, batch_u]
        }
        GPUOperation::MatMul3D1D_F32 | GPUOperation::MatMul1D3D_F32 => {
            [dispatch_x as u64, dispatch_y as u64, dispatch_z as u64]
        }
        _ => [dispatch_x as u64, dispatch_y as u64, dispatch_z as u64],
    };

    gpu.dispatch(command_buffer, local_size, work_size);

    Ok(())
}
