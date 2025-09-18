#[repr(C)]
pub struct InitUniformPushConstants {
    pub total_elements: u32,
    pub seed: u32,
    pub min_val: f32,
    pub max_val: f32,
}
