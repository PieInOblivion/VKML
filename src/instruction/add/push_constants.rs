#[repr(C)]
pub struct AddPushConstants {
    pub rank: u32,
    pub pad: u32,
    pub total: u32,
    pub dims: [u32; 8],
    pub strides_a: [u32; 8],
    pub strides_b: [u32; 8],
}
