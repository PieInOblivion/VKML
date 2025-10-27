use bytemuck::{try_cast_slice, try_cast_slice_mut};

pub fn f32_f32_cpu(
    strides_src: Vec<usize>,
    dst_dims: Vec<i64>,
    src_bytes: &[u8],
    dst_bytes: &mut [u8],
) {
    let num_elements: usize = dst_dims.iter().map(|d| *d as usize).product();
    let src_f32: &[f32] = try_cast_slice(src_bytes)
        .expect("src byte slice cannot be cast to f32 slice (alignment/length mismatch)");
    let dst_f32: &mut [f32] = try_cast_slice_mut(dst_bytes)
        .expect("dst byte slice cannot be cast to f32 slice (alignment/length mismatch)");

    assert_eq!(dst_f32.len(), num_elements, "dst buffer size mismatch");

    let rank = dst_dims.len();
    let dims_usize: Vec<usize> = dst_dims.iter().map(|d| *d as usize).collect();

    // odometer indices for efficient iteration
    let mut idxs = vec![0usize; rank];
    let mut off_src: usize = 0;

    for dst_slot in dst_f32.iter_mut().take(num_elements) {
        *dst_slot = src_f32[off_src];

        // increment odometer
        for d in (0..rank).rev() {
            idxs[d] += 1;
            off_src = off_src.wrapping_add(strides_src[d]);

            if idxs[d] < dims_usize[d] {
                break; // no carry, continue main loop
            } else {
                // carry: reset this index and subtract the wrapped amount from offset
                idxs[d] = 0;
                off_src = off_src.wrapping_sub(strides_src[d] * dims_usize[d]);
            }
        }
    }
}
