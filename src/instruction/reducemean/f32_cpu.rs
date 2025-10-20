use bytemuck::{try_cast_slice, try_cast_slice_mut};

use crate::TensorDesc;

/// Compute ReduceMean for f32 inputs on CPU.
///
/// Parameters:
/// - `src_bytes`: source tensor bytes
/// - `src_dims`: source tensor dims
/// - `axes_vec`: axes to reduce (as i64)
/// - `keep`: whether to keep reduced dims as 1
/// - `dst_ptr`: destination byte buffer (must be sized to dst_dims)
pub fn f32_cpu(
    src_bytes: &[u8],
    src_dims: &[i64],
    axes_vec: &[i64],
    keep: bool,
    dst_ptr: &mut [u8],
) {
    // Cast source to f32 slice
    let src_f32: &[f32] = try_cast_slice(src_bytes)
        .expect("src byte slice cannot be cast to f32 slice (alignment/length mismatch)");

    // Compute output dims
    let mut out_dims: Vec<i64> = Vec::new();
    for (i, &d) in src_dims.iter().enumerate() {
        if axes_vec.contains(&(i as i64)) {
            if keep {
                out_dims.push(1);
            }
        } else {
            out_dims.push(d);
        }
    }
    if !keep && out_dims.is_empty() {
        out_dims.push(1);
    }

    // Cast dst once
    let dst_f32: &mut [f32] = try_cast_slice_mut(dst_ptr)
        .expect("dst byte slice cannot be cast to f32 slice (alignment/length mismatch)");

    // Prepare output buffer
    let out_num: usize = dst_f32.len();
    let mut out_vals = vec![0f32; out_num];

    // compute strides
    let src_strides = TensorDesc::compute_strides(src_dims);
    let out_strides = TensorDesc::compute_strides(&out_dims);

    // accumulate
    for (idx, &src_val) in src_f32.iter().enumerate() {
        // compute multi-index of src
        let mut rem = idx;
        let mut out_idx = 0usize;
        for (dim_i, &s) in src_strides.iter().enumerate() {
            let coord = rem / s;
            rem %= s;
            if !axes_vec.contains(&(dim_i as i64)) {
                out_idx += coord
                    * out_strides[dim_i - axes_vec.iter().filter(|&&a| a < dim_i as i64).count()];
            }
        }
        out_vals[out_idx] += src_val;
    }

    // divide by reduction size per output element
    let mut reduction_size = 1usize;
    for &a in axes_vec {
        reduction_size *= src_dims[a as usize] as usize;
    }
    for v in out_vals.iter_mut() {
        *v /= reduction_size as f32;
    }

    // write to dst (use the cast dst_f32 slice)
    for (i, &val) in out_vals.iter().enumerate() {
        dst_f32[i] = val;
    }
}
