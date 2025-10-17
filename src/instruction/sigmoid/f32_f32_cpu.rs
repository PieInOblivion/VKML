use bytemuck::{try_cast_slice, try_cast_slice_mut};

// Broadcast-aware sigmoid helper: dst = 1/(1+exp(-src))
pub fn f32_f32_cpu(
    strides_a: Vec<usize>,
    _strides_b: Vec<usize>,
    dst_dims: Vec<i64>,
    src_bytes: &[u8],
    _rhs_bytes: &[u8],
    dst_ptr: &mut [u8],
) {
    let num_elements: usize = dst_dims.iter().map(|d| *d as usize).product();

    let src_f32: &[f32] = try_cast_slice(src_bytes)
        .expect("src byte slice cannot be cast to f32 slice (alignment/length mismatch)");
    let dst_f32: &mut [f32] = try_cast_slice_mut(dst_ptr)
        .expect("dst byte slice cannot be cast to f32 slice (alignment/length mismatch)");

    assert_eq!(dst_f32.len(), num_elements, "dst buffer size mismatch");

    let rank = dst_dims.len();
    let dims_usize: Vec<usize> = dst_dims.iter().map(|d| *d as usize).collect();

    let mut idxs = vec![0usize; rank];

    let mut off_a: usize = 0;

    for dst_slot in dst_f32.iter_mut().take(num_elements) {
        let x = src_f32[off_a];
        *dst_slot = 1.0 / (1.0 + (-x).exp());

        for d in (0..rank).rev() {
            idxs[d] += 1;
            off_a = off_a.wrapping_add(strides_a[d]);

            if idxs[d] < dims_usize[d] {
                break;
            } else {
                idxs[d] = 0;
                off_a = off_a.wrapping_sub(strides_a[d] * dims_usize[d]);
            }
        }
    }
}
