use bytemuck::{try_cast_slice, try_cast_slice_mut};

use crate::TensorDesc;

/// A simple single-threaded N-D convolution for f32 tensors.
pub fn f32_f32_f32_f32_cpu(
    src_dims: Vec<usize>,
    weight_dims: Vec<usize>,
    dst_dims: Vec<usize>,
    src_bytes: &[u8],
    weight_bytes: &[u8],
    bias_bytes: Option<&[u8]>,
    dst_ptr: &mut [u8],
    stride: Vec<usize>,
    pads_begin: Vec<usize>,
    dilation: Vec<usize>,
    group: usize,
) {
    // Cast to f32 slices
    let src_f: &[f32] = try_cast_slice(src_bytes).expect("src bytes not f32");
    let weight_f: &[f32] = try_cast_slice(weight_bytes).expect("weight bytes not f32");
    let dst_f: &mut [f32] = try_cast_slice_mut(dst_ptr).expect("dst bytes not f32");

    // weight layout: [M, C/group, k1, k2, ...]
    // src layout: [N, C, D1, D2, ...]
    // dst layout: [N, M, O1, O2, ...]

    let n = src_dims[0];
    let c = src_dims[1];
    let m = weight_dims[0];

    // Validate group configuration
    if group == 0 || !c.is_multiple_of(group) || !m.is_multiple_of(group) {
        panic!(
            "f32_cpu: unsupported group configuration: group={}, C={}, M={}",
            group, c, m
        );
    }
    let m_per_group = m / group;
    let c_per_group = c / group;

    let spatial_rank = src_dims.len() - 2;

    // Compute strides for indexing
    let src_strides =
        TensorDesc::compute_strides(&src_dims.iter().map(|d| *d as i64).collect::<Vec<_>>());
    let dst_strides =
        TensorDesc::compute_strides(&dst_dims.iter().map(|d| *d as i64).collect::<Vec<_>>());
    let weight_strides =
        TensorDesc::compute_strides(&weight_dims.iter().map(|d| *d as i64).collect::<Vec<_>>());

    // Helper to compute linear offset
    let offset = |idxs: &[usize], strides: &[usize]| -> usize {
        idxs.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
    };

    // Bias as f32 slice if present
    let bias_f: Option<&[f32]> = bias_bytes.map(|b| try_cast_slice(b).expect("bias bytes not f32"));

    // For each batch n and output channel m and spatial location, compute convolution
    // We'll iterate over N, M, and spatial output positions, and accumulate over input channels and kernel positions

    // Precompute kernel spatial shape and number of kernel elements
    let kernel_spatial: Vec<usize> = weight_dims[2..].to_vec();
    let kernel_elems: usize = kernel_spatial.iter().product();

    // Iterate
    for ni in 0..n {
        for mi in 0..m {
            // For each output spatial index, represented as a multi-index
            let out_spatial_counts = &dst_dims[2..];
            let mut out_index = vec![0usize; spatial_rank];
            loop {
                // compute dst linear index
                let mut dst_idxs = vec![0usize; 2 + spatial_rank];
                dst_idxs[0] = ni;
                dst_idxs[1] = mi;
                for (i, &v) in out_index.iter().enumerate() {
                    dst_idxs[2 + i] = v;
                }
                let dst_off = offset(&dst_idxs, &dst_strides);

                let mut acc: f32 = 0.0;

                // determine channel range for this output channel's group
                let group_id = mi / m_per_group;
                let c_start = group_id * c_per_group;
                let c_end = c_start + c_per_group;

                // accumulate over input channels in the same group and kernel positions
                for ci in c_start..c_end {
                    for k_idx in 0..kernel_elems {
                        // convert k_idx to multi-index over kernel_spatial
                        let mut rem = k_idx;
                        let mut k_multi = vec![0usize; spatial_rank];
                        for d in (0..spatial_rank).rev() {
                            let dim = kernel_spatial[d];
                            k_multi[d] = rem % dim;
                            rem /= dim;
                        }

                        // compute input spatial position: in_pos = out_pos*stride - pad_begin + k*dilation
                        let mut src_idxs = vec![0usize; 2 + spatial_rank];
                        src_idxs[0] = ni;
                        src_idxs[1] = ci;
                        let mut in_bounds = true;
                        for (i, &out_v) in out_index.iter().enumerate() {
                            let o = out_v as isize;
                            let s = stride[i] as isize;
                            let p_b = pads_begin[i] as isize;
                            let dil = dilation[i] as isize;
                            let kpos = k_multi[i] as isize;
                            let in_pos = o * s - p_b + kpos * dil;
                            if in_pos < 0 || in_pos >= src_dims[2 + i] as isize {
                                in_bounds = false;
                                break;
                            }
                            src_idxs[2 + i] = in_pos as usize;
                        }

                        if !in_bounds {
                            continue;
                        }

                        // linear offsets
                        let src_off = offset(&src_idxs, &src_strides);
                        // weight index: [mi, ci_in_group, k_multi...]
                        let mut w_idxs = vec![0usize; 2 + spatial_rank];
                        w_idxs[0] = mi;
                        w_idxs[1] = ci - c_start; // channel index within the group
                        for (i, &km) in k_multi.iter().enumerate() {
                            w_idxs[2 + i] = km;
                        }
                        let w_off = offset(&w_idxs, &weight_strides);

                        acc += src_f[src_off] * weight_f[w_off];
                    }
                }

                // add bias
                if let Some(bf) = bias_f {
                    acc += bf[mi];
                }

                dst_f[dst_off] = acc;

                // increment out_index
                let mut carry = 1usize;
                for i in (0..spatial_rank).rev() {
                    out_index[i] += carry;
                    if out_index[i] >= out_spatial_counts[i] {
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
