use bytemuck::{try_cast_slice, try_cast_slice_mut};

// CPU softmax helper for f32. Currently supports softmax along the last dimension
// but uses canonical tensor byte-slice casting and performs numerically stable softmax per-batch.
pub fn f32_cpu(dst_dims: &[i64], dim: usize, src_bytes: &[u8], dst_ptr: &mut [u8]) {
    // verify dims
    assert!(!dst_dims.is_empty(), "dst dims empty");
    assert_eq!(
        dim,
        dst_dims.len() - 1,
        "Softmax helper currently only supports the last dimension"
    );

    let num_elements: usize = dst_dims.iter().map(|d| *d as usize).product();

    let src_f32: &[f32] = try_cast_slice(src_bytes)
        .expect("src byte slice cannot be cast to f32 slice (alignment/length mismatch)");
    let dst_f32: &mut [f32] = try_cast_slice_mut(dst_ptr)
        .expect("dst byte slice cannot be cast to f32 slice (alignment/length mismatch)");

    assert_eq!(dst_f32.len(), num_elements, "dst buffer size mismatch");

    let feature_size = dst_dims[dim] as usize;
    let batch_size = num_elements / feature_size;

    for b in 0..batch_size {
        let offset = b * feature_size;

        // find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..feature_size {
            let v = src_f32[offset + i];
            if v > max_val {
                max_val = v;
            }
        }

        // compute exponentials and sum
        let mut sum = 0.0f32;
        for i in 0..feature_size {
            let exp_val = (src_f32[offset + i] - max_val).exp();
            dst_f32[offset + i] = exp_val;
            sum += exp_val;
        }

        // normalize
        if sum == 0.0 {
            // avoid div-by-zero; fallback to uniform distribution
            let inv = 1.0 / feature_size as f32;
            for i in 0..feature_size {
                dst_f32[offset + i] = inv;
            }
        } else {
            for i in 0..feature_size {
                dst_f32[offset + i] /= sum;
            }
        }
    }
}
