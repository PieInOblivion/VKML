#[repr(C)]
pub struct InitHePushConstants {
    pub total_elements: u32,
    pub fan_in: u32,
    pub seed: u32,
    pub gain: f32,
}
