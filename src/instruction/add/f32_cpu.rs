use crate::tensor::desc::TensorDesc;

pub fn f32_cpu(
    strides_a: Vec<usize>,
    strides_b: Vec<usize>,
    dst_dims: Vec<i64>,
    src1_bytes: &[u8],
    src2_bytes: &[u8],
    dst_ptr: &mut [u8],
) {
    for i in 0..dst_ptr.len() {
        let idxs = TensorDesc::unravel(i, &dst_dims);
        let off1 = TensorDesc::offset(&idxs, &strides_a);
        let off2 = TensorDesc::offset(&idxs, &strides_b);
        dst_ptr[i] = src1_bytes[off1] + src2_bytes[off2];
    }
}
