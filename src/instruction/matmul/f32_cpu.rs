use bytemuck::{try_cast_slice, try_cast_slice_mut};

pub fn f32_cpu(
    src1_dims: Vec<usize>,
    src2_dims: Vec<usize>,
    dst_dims: Vec<usize>,
    src1_bytes: &[u8],
    src2_bytes: &[u8],
    dst_bytes: &mut [u8],
) {
    let src1_data: &[f32] =
        try_cast_slice(src1_bytes).expect("src1 bytes cannot be cast to f32 slice");
    let src2_data: &[f32] =
        try_cast_slice(src2_bytes).expect("src2 bytes cannot be cast to f32 slice");
    let dst_data: &mut [f32] =
        try_cast_slice_mut(dst_bytes).expect("dst bytes cannot be cast to f32 slice");

    // Zero initialize result
    for val in dst_data.iter_mut() {
        *val = 0.0
    }

    // Handle special cases for 1D tensors
    let (effective_src1_dims, effective_src2_dims) = match (src1_dims.len(), src2_dims.len()) {
        (1, 1) => panic!("MatMul between two 1D tensors is not supported"),
        (1, _) => {
            let mut dims = Vec::with_capacity(src1_dims.len() + 1);
            dims.push(1);
            dims.extend_from_slice(&src1_dims);
            (dims, src2_dims.clone())
        }
        (_, 1) => {
            let mut dims = Vec::with_capacity(src2_dims.len() + 1);
            dims.extend_from_slice(&src2_dims);
            dims.push(1);
            (src1_dims.clone(), dims)
        }
        _ => (src1_dims.clone(), src2_dims.clone()),
    };

    // Extract core matrix dimensions
    assert!(
        effective_src1_dims.len() >= 2 && effective_src2_dims.len() >= 2,
        "After adjustment, tensors must have at least 2 dimensions for MatMul"
    );

    let src1_matrix_dims = &effective_src1_dims[effective_src1_dims.len() - 2..];
    let src2_matrix_dims = &effective_src2_dims[effective_src2_dims.len() - 2..];

    let m = src1_matrix_dims[0];
    let k1 = src1_matrix_dims[1];
    let k2 = src2_matrix_dims[0];
    let n = src2_matrix_dims[1];

    assert_eq!(
        k1, k2,
        "Inner dimensions don't match for matrix multiplication: {} vs {}",
        k1, k2
    );

    // Extract batch dimensions
    let src1_batch_dims = &effective_src1_dims[..effective_src1_dims.len() - 2];
    let src2_batch_dims = &effective_src2_dims[..effective_src2_dims.len() - 2];

    // Validate batch dimensions
    let batch_dims = if src1_batch_dims.is_empty() {
        src2_batch_dims.to_vec()
    } else if src2_batch_dims.is_empty() || src1_batch_dims == src2_batch_dims {
        src1_batch_dims.to_vec()
    } else {
        panic!(
            "Incompatible batch dimensions: {:?} vs {:?}",
            src1_batch_dims, src2_batch_dims
        );
    };

    let mut expected_output_dims = batch_dims.clone();
    expected_output_dims.push(m);
    expected_output_dims.push(n);

    let expected_output_dims = match (src1_dims.len(), src2_dims.len()) {
        (1, _) => expected_output_dims[expected_output_dims.len() - 2..].to_vec(),
        (_, 1) => {
            let mut dims = batch_dims.clone();
            dims.push(m);
            dims
        }
        _ => expected_output_dims,
    };

    assert_eq!(
        *dst_dims, expected_output_dims,
        "Output dimensions mismatch: expected {:?}, got {:?}",
        expected_output_dims, dst_dims
    );

    // strides
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

    let total_batches = batch_dims.iter().product::<usize>().max(1);

    // Reuse a single buffer for batch indices to avoid allocating per-batch.
    let mut batch_indices = vec![0usize; batch_dims.len()];

    let compute_indices_inplace = |flat_idx: usize, dims: &[usize], out: &mut [usize]| {
        if out.len() != dims.len() {
            // Safety: caller should ensure lengths match, but guard just in case.
            for v in out.iter_mut() {
                *v = 0;
            }
            return;
        }
        let mut remaining = flat_idx;
        for i in (0..dims.len()).rev() {
            out[i] = remaining % dims[i];
            remaining /= dims[i];
        }
    };

    let calculate_offset = |indices: &[usize], strides: &[usize]| -> usize {
        let mut offset = 0usize;
        for (i, &idx) in indices.iter().enumerate() {
            offset += idx * strides[i];
        }
        offset
    };

    for batch_idx in 0..total_batches {
        // Compute batch offsets using the reusable batch_indices buffer.
        if !batch_dims.is_empty() {
            compute_indices_inplace(batch_idx, &batch_dims, &mut batch_indices);
        }

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
            calculate_offset(&batch_indices, &dst_strides[..dst_dims.len() - 2])
        };

        // Cache commonly used indices and flags to reduce repeated indexing and len() calls
        let a_is_1d = src1_dims.len() == 1;
        let b_is_1d = src2_dims.len() == 1;
        let eff_a_len = effective_src1_dims.len();
        let eff_b_len = effective_src2_dims.len();
        let a_row_stride = src1_strides[eff_a_len - 2];
        let a_col_stride = src1_strides[eff_a_len - 1];
        let b_row_stride = src2_strides[eff_b_len - 2];
        let b_col_stride = src2_strides[eff_b_len - 1];
        let dst_row_stride = dst_strides[dst_dims.len() - 2];
        let dst_col_stride = dst_strides[dst_dims.len() - 1];

        for i in 0..m {
            // Precompute row base offsets where possible
            let src1_row_base = if a_is_1d {
                0
            } else {
                src1_batch_offset + i * a_row_stride
            };
            let dst_row_base = dst_batch_offset + i * dst_row_stride;

            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k1 {
                    let src1_idx = if a_is_1d {
                        kk
                    } else {
                        src1_row_base + kk * a_col_stride
                    };
                    let src2_idx = if b_is_1d {
                        kk
                    } else {
                        src2_batch_offset + kk * b_row_stride + j * b_col_stride
                    };

                    sum += src1_data[src1_idx] * src2_data[src2_idx];
                }

                let dst_idx = if a_is_1d && dst_dims.len() == 1 {
                    j
                } else if b_is_1d && dst_dims.len() == 1 {
                    i
                } else {
                    dst_row_base + j * dst_col_stride
                };

                dst_data[dst_idx] = sum;
            }
        }
    }
}
