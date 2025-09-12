use crate::utils::math::normal_sample;
use bytemuck::try_cast_slice_mut;

pub fn f32_cpu(fan_in: usize, dst_dims: Vec<i64>, dst_ptr: &mut [u8]) {
    let num_elements: usize = dst_dims.iter().map(|d| *d as usize).product();

    let dst_f32: &mut [f32] = try_cast_slice_mut(dst_ptr)
        .expect("dst byte slice cannot be cast to f32 slice (alignment/length mismatch)");

    assert_eq!(dst_f32.len(), num_elements, "dst buffer size mismatch");

    let std_dev = (2.0f32 / fan_in as f32).sqrt();

    for i in 0..num_elements {
        let v = normal_sample(0.0, std_dev);
        dst_f32[i] = v;
    }
}
