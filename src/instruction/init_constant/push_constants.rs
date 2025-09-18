#[repr(C)]
pub struct InitConstantPushConstants {
    pub elem_size: u32,
    pub value_lo: u32,
    pub value_hi: u32,
}
