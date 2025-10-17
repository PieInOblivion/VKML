#[repr(C)]
pub struct GemmPushConstants {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub stride_a0: u32,
    pub stride_a1: u32,
    pub stride_b0: u32,
    pub stride_b1: u32,
    pub stride_y0: u32,
    pub stride_y1: u32,
    pub trans_a: u32,
    pub trans_b: u32,
    pub alpha: u32, // f32 as raw bits
    pub beta: u32,  // f32 as raw bits
    pub has_c: u32,
}
