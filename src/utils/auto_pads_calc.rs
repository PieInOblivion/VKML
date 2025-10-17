use crate::instruction::AutoPad;
use crate::tensor::TensorDesc;

/// Compute pads_begin and pads_end following ONNX semantics.
///
/// Parameters mirror the instruction fields: `auto_pad`, explicit `pads` (may be empty),
/// `kernel_shape`, `strides`, `dilations`, and the `src_desc` for input spatial sizes.
pub fn calc_begin_and_end_pads(
    auto_pad: AutoPad,
    pads: &[usize],
    kernel_shape: &[usize],
    strides: &[usize],
    dilations: &[usize],
    src_desc: &TensorDesc,
) -> (Vec<usize>, Vec<usize>) {
    let spatial_rank = if src_desc.ndim() >= 2 {
        src_desc.ndim() - 2
    } else {
        0
    };

    let mut stride_vec: Vec<usize> = Vec::with_capacity(spatial_rank);
    let mut dilation_vec: Vec<usize> = Vec::with_capacity(spatial_rank);
    let mut kernel_vec: Vec<usize> = Vec::with_capacity(spatial_rank);
    for i in 0..spatial_rank {
        stride_vec.push(strides.get(i).copied().unwrap_or(1));
        dilation_vec.push(dilations.get(i).copied().unwrap_or(1));
        kernel_vec.push(kernel_shape.get(i).copied().unwrap_or(1));
    }

    let mut pads_begin: Vec<usize> = vec![0; spatial_rank];
    let mut pads_end: Vec<usize> = vec![0; spatial_rank];

    if pads.len() >= spatial_rank * 2 {
        pads_begin[..spatial_rank].copy_from_slice(&pads[..spatial_rank]);
        pads_end[..spatial_rank].copy_from_slice(&pads[spatial_rank..(spatial_rank * 2)]);
    } else if pads.len() == spatial_rank {
        pads_begin[..spatial_rank].copy_from_slice(&pads[..spatial_rank]);
        pads_end[..spatial_rank].copy_from_slice(&pads[..spatial_rank]);
    } else if auto_pad != AutoPad::NotSet {
        for i in 0..spatial_rank {
            let in_i = src_desc.dims()[i + 2];
            let k = kernel_vec[i] as i64;
            let s = stride_vec[i] as i64;
            let d = dilation_vec[i] as i64;

            if auto_pad == AutoPad::Valid {
                pads_begin[i] = 0;
                pads_end[i] = 0;
            } else {
                let out = (in_i + s - 1) / s; // ceil
                let pad_needed = ((out - 1) * s + d * (k - 1) + 1) - in_i;
                let pad_needed = if pad_needed > 0 { pad_needed } else { 0 } as usize;
                if auto_pad == AutoPad::SameUpper {
                    pads_begin[i] = pad_needed / 2;
                    pads_end[i] = pad_needed - pads_begin[i];
                } else {
                    pads_end[i] = pad_needed / 2;
                    pads_begin[i] = pad_needed - pads_end[i];
                }
            }
        }
    }

    (pads_begin, pads_end)
}
