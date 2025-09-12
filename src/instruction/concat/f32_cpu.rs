use bytemuck::{try_cast_slice, try_cast_slice_mut};

pub fn f32_cpu(
    src_bytes_vec: &[&[u8]],
    src_dims_vec: &[Vec<i64>],
    axis: usize,
    dst_dims: &[i64],
    dst_ptr: &mut [u8],
) {
    assert!(
        !src_dims_vec.is_empty(),
        "no source tensors provided to concat"
    );

    let rank = dst_dims.len();
    assert!(axis < rank, "concat axis out of bounds");

    // Validate that all source ranks match destination rank and that dims
    // other than 'axis' are equal across sources.
    for src_dims in src_dims_vec {
        assert_eq!(src_dims.len(), rank, "source rank mismatch");
        for (i, &d) in src_dims.iter().enumerate() {
            if i == axis {
                continue;
            }
            assert_eq!(d, dst_dims[i], "non-concat dimension mismatch");
        }
    }

    // Compute sizes used for copying contiguous blocks.
    // outer = product of dims[0..axis]
    // inner = product of dims[axis+1..]
    let outer: usize = dst_dims[0..axis].iter().map(|d| *d as usize).product();
    let inner: usize = if axis + 1 <= rank - 1 {
        dst_dims[axis + 1..].iter().map(|d| *d as usize).product()
    } else {
        1usize
    };

    // Cast dst once
    let dst_f32: &mut [f32] = try_cast_slice_mut(dst_ptr)
        .expect("dst byte slice cannot be cast to f32 slice (alignment/length mismatch)");
    let total_elements: usize = dst_dims.iter().map(|d| *d as usize).product();
    assert_eq!(dst_f32.len(), total_elements, "dst buffer size mismatch");

    // Cast each source once and compute its axis length
    let mut src_slices: Vec<&[f32]> = Vec::with_capacity(src_bytes_vec.len());
    let mut src_axis_lens: Vec<usize> = Vec::with_capacity(src_bytes_vec.len());
    for (i, &b) in src_bytes_vec.iter().enumerate() {
        let s: &[f32] = try_cast_slice(b)
            .expect("src byte slice cannot be cast to f32 slice (alignment/length mismatch)");
        let axis_len = src_dims_vec[i][axis] as usize;
        src_slices.push(s);
        src_axis_lens.push(axis_len);
    }

    // For each outer index, we copy blocks across sources along the axis.
    // For each source, the contiguous block size = axis_len * inner.
    let mut dst_outer_offset = 0usize; // in f32 elements
    for outer_idx in 0..outer {
        // For each source, copy its block for this outer index
        let mut dst_axis_offset = 0usize; // offset along axis in this outer slice
        for (src_idx, &axis_len) in src_axis_lens.iter().enumerate() {
            let src = src_slices[src_idx];

            // Compute starting offsets in elements
            let src_block_start = outer_idx * axis_len * inner;
            let src_block_end = src_block_start + axis_len * inner;

            let dst_block_start = dst_outer_offset + dst_axis_offset * inner;
            let dst_block_end = dst_block_start + axis_len * inner;

            dst_f32[dst_block_start..dst_block_end]
                .copy_from_slice(&src[src_block_start..src_block_end]);

            dst_axis_offset += axis_len;
        }

        dst_outer_offset += dst_dims[axis] as usize * inner;
    }
}
