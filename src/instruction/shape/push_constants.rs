#[repr(C)]
pub struct ShapePushConstants {
    pub slice_len: u32,
    pub start: u32,
    pub pad: u32,
    // dims split into low/high 32-bit words for each 64-bit value; max 8 dims
    pub dims_lo: [u32; 8],
    pub dims_hi: [u32; 8],
}
