use bytemuck::{try_cast_slice, try_cast_slice_mut};

use crate::TensorDesc;

/// N-D max pooling f32 CPU implementation. No indices are produced.
pub fn f32_f32_cpu(
    src_dims: Vec<usize>,
    dst_dims: Vec<usize>,
    src_bytes: &[u8],
    dst_ptr: &mut [u8],
    kernel: Vec<usize>,
    stride: Vec<usize>,
    pads_begin: Vec<usize>,
    dilation: Vec<usize>,
) {
    let src_f: &[f32] = try_cast_slice(src_bytes).expect("src bytes not f32");
    let dst_f: &mut [f32] = try_cast_slice_mut(dst_ptr).expect("dst bytes not f32");

    // Layouts: src [N, C, D1..], dst [N, C, O1..]
    assert!(
        src_dims.len() >= 2 && dst_dims.len() >= 2,
        "MaxPool: dims too small"
    );

    let n = src_dims[0];
    let c = src_dims[1];
    let spatial_rank = src_dims.len() - 2;

    // compute strides
    let src_strides =
        TensorDesc::compute_strides(&src_dims.iter().map(|d| *d as i64).collect::<Vec<_>>());
    let dst_strides =
        TensorDesc::compute_strides(&dst_dims.iter().map(|d| *d as i64).collect::<Vec<_>>());

    let offset = |idxs: &[usize], strides: &[usize]| -> usize {
        idxs.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
    };

    // For each (n, c, out_spatial...) compute max over kernel window
    for ni in 0..n {
        for ci in 0..c {
            // iterate over output positions via mixed-radix counting
            let out_counts = &dst_dims[2..];
            let mut out_index = vec![0usize; spatial_rank];
            loop {
                // compute dst offset
                let mut dst_idxs = vec![0usize; 2 + spatial_rank];
                dst_idxs[0] = ni;
                dst_idxs[1] = ci;
                for (i, &v) in out_index.iter().enumerate() {
                    dst_idxs[2 + i] = v;
                }
                let dst_off = offset(&dst_idxs, &dst_strides);

                // scan kernel positions and compute max
                // If a window contains no valid input positions (fully padded) we write 0.0
                let mut max_val: f32 = f32::NEG_INFINITY;
                let mut found = false;

                // nested loops over kernel elements via mixed radix
                let mut k_multi = vec![0usize; spatial_rank];
                loop {
                    // compute input positions
                    let mut src_idxs = vec![0usize; 2 + spatial_rank];
                    src_idxs[0] = ni;
                    src_idxs[1] = ci;
                    let mut in_bounds = true;
                    for (i, &out_v) in out_index.iter().enumerate() {
                        let o = out_v as isize;
                        let s = stride[i] as isize;
                        let p = pads_begin.get(i).copied().unwrap_or(0) as isize;
                        let dil = dilation[i] as isize;
                        let kpos = k_multi[i] as isize;
                        let in_pos = o * s - p + kpos * dil;
                        if in_pos < 0 || in_pos >= src_dims[2 + i] as isize {
                            in_bounds = false;
                            break;
                        }
                        src_idxs[2 + i] = in_pos as usize;
                    }

                    if in_bounds {
                        let src_off = offset(&src_idxs, &src_strides);
                        let val = src_f[src_off];
                        if val > max_val {
                            max_val = val;
                        }
                        found = true;
                    }

                    // increment k_multi
                    let mut carry = 1usize;
                    for i in (0..spatial_rank).rev() {
                        k_multi[i] += carry;
                        if k_multi[i] >= kernel[i] {
                            k_multi[i] = 0;
                            carry = 1;
                        } else {
                            carry = 0;
                            break;
                        }
                    }
                    if carry == 1 {
                        break;
                    }
                }

                // If no valid in-bounds values found (fully padded window), write 0.0.
                // Otherwise write the computed max value.
                if found {
                    dst_f[dst_off] = max_val;
                } else {
                    dst_f[dst_off] = 0.0;
                }

                // increment out_index
                let mut carry = 1usize;
                for i in (0..spatial_rank).rev() {
                    out_index[i] += carry;
                    if out_index[i] >= out_counts[i] {
                        out_index[i] = 0;
                        carry = 1;
                    } else {
                        carry = 0;
                        break;
                    }
                }
                if carry == 1 {
                    break;
                }
            }
        }
    }
}
