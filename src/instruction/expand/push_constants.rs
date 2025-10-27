#[repr(C)]
pub struct ExpandPushConstants {
    pub rank: u32,
    pub pad: u32,
    pub total: u32,
    pub dims: [u32; 8],
    pub strides_src: [u32; 8],
}
