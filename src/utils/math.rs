use rand::distr::{Distribution, Uniform};
use std::f32::consts::PI;

/// Box-Muller transform to generate normal distribution
pub fn normal_sample(mean: f32, std_dev: f32) -> f32 {
    let mut rng = rand::rng();
    let uniform = Uniform::new(0.0f32, 1.0);

    let u1 = uniform.unwrap().sample(&mut rng);
    let u2 = uniform.unwrap().sample(&mut rng);

    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    mean + std_dev * z
}
