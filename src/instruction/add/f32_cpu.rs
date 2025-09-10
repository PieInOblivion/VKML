use bytemuck::{try_cast_slice, try_cast_slice_mut};

pub fn f32_cpu(
    strides_a: Vec<usize>,
    strides_b: Vec<usize>,
    dst_dims: Vec<i64>,
    src1_bytes: &[u8],
    src2_bytes: &[u8],
    dst_ptr: &mut [u8],
) {
    // number of elements in destination
    let num_elements: usize = dst_dims.iter().map(|d| *d as usize).product();

    // cast raw bytes to f32 slices (panic if cast fails as requested)
    let src1_f32: &[f32] = try_cast_slice(src1_bytes)
        .expect("src1 byte slice cannot be cast to f32 slice (alignment/length mismatch)");
    let src2_f32: &[f32] = try_cast_slice(src2_bytes)
        .expect("src2 byte slice cannot be cast to f32 slice (alignment/length mismatch)");
    let dst_f32: &mut [f32] = try_cast_slice_mut(dst_ptr)
        .expect("dst byte slice cannot be cast to f32 slice (alignment/length mismatch)");

    assert_eq!(dst_f32.len(), num_elements, "dst buffer size mismatch");

    let rank = dst_dims.len();
    let dims_usize: Vec<usize> = dst_dims.iter().map(|d| *d as usize).collect();

    // odometer indices for efficient iteration without division/modulo per element
    let mut idxs = vec![0usize; rank];

    let mut off_a: usize = 0;
    let mut off_b: usize = 0;

    for i in 0..num_elements {
        dst_f32[i] = src1_f32[off_a] + src2_f32[off_b];

        // increment odometer
        for d in (0..rank).rev() {
            idxs[d] += 1;
            off_a = off_a.wrapping_add(strides_a[d]);
            off_b = off_b.wrapping_add(strides_b[d]);

            if idxs[d] < dims_usize[d] {
                break; // no carry, continue main loop
            } else {
                // carry: reset this index and subtract the wrapped amount from offsets
                idxs[d] = 0;
                off_a = off_a.wrapping_sub(strides_a[d] * dims_usize[d]);
                off_b = off_b.wrapping_sub(strides_b[d] * dims_usize[d]);
                // continue to next more significant digit
            }
        }
    }
}
