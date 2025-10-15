use bytemuck::try_cast_slice_mut;
use rand::distr::{Distribution, Uniform};

pub fn f32_cpu(fan_in: usize, fan_out: usize, dst_dims: Vec<i64>, dst_ptr: &mut [u8]) {
    let num_elements: usize = dst_dims.iter().map(|d| *d as usize).product();

    let dst_f32: &mut [f32] = try_cast_slice_mut(dst_ptr)
        .expect("dst byte slice cannot be cast to f32 slice (alignment/length mismatch)");

    assert_eq!(dst_f32.len(), num_elements, "dst buffer size mismatch");

    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let dist = Uniform::new(-limit, limit);
    let mut rng = rand::rng();

    for dst_slot in dst_f32.iter_mut().take(num_elements) {
        let v: f32 = dist.unwrap().sample(&mut rng);
        *dst_slot = v;
    }
}
