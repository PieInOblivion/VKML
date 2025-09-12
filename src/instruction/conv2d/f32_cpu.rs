use bytemuck::{try_cast_slice, try_cast_slice_mut};

pub fn f32_cpu(
    src_dims: Vec<usize>,
    weight_dims: Vec<usize>,
    dst_dims: Vec<usize>,
    src_bytes: &[u8],
    weight_bytes: &[u8],
    bias_bytes: Option<&[u8]>,
    dst_ptr: &mut [u8],
    stride: (usize, usize),
    padding: (usize, usize),
) {
    let src_f32: &[f32] = try_cast_slice(src_bytes).expect("src bytes not f32");
    let weight_f32: &[f32] = try_cast_slice(weight_bytes).expect("weight bytes not f32");
    let dst_f32: &mut [f32] = try_cast_slice_mut(dst_ptr).expect("dst bytes not f32");

    let bias_f32: Option<&[f32]> = match bias_bytes {
        Some(b) => Some(try_cast_slice(b).expect("bias bytes not f32")),
        None => None,
    };

    // Unpack dims: expected layout [batch, channels, height, width]
    assert_eq!(src_dims.len(), 4);
    assert_eq!(weight_dims.len(), 4);
    assert_eq!(dst_dims.len(), 4);

    let batch = src_dims[0];
    let in_channels = src_dims[1];
    let in_h = src_dims[2];
    let in_w = src_dims[3];

    let out_channels = weight_dims[0];
    let filter_in_channels = weight_dims[1];
    let k_h = weight_dims[2];
    let k_w = weight_dims[3];

    let out_h = dst_dims[2];
    let out_w = dst_dims[3];

    assert_eq!(in_channels, filter_in_channels);

    for b in 0..batch {
        for oc in 0..out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;

                    for ic in 0..in_channels {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                // compute input coords with padding and stride
                                let ih_pos = oh * stride.0 + kh;
                                let iw_pos = ow * stride.1 + kw;

                                // subtract padding
                                if ih_pos < padding.0 || iw_pos < padding.1 {
                                    continue;
                                }

                                let ih = ih_pos - padding.0;
                                let iw = iw_pos - padding.1;

                                if ih < in_h && iw < in_w {
                                    let in_idx = ((b * in_channels + ic) * in_h + ih) * in_w + iw;
                                    let w_idx = ((oc * in_channels + ic) * k_h + kh) * k_w + kw;
                                    sum += src_f32[in_idx] * weight_f32[w_idx];
                                }
                            }
                        }
                    }

                    if let Some(bias) = bias_f32 {
                        sum += bias[oc];
                    }

                    let out_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
                    dst_f32[out_idx] = sum;
                }
            }
        }
    }
}
