#[repr(C)]
pub struct SoftmaxPushConstants {
    pub batch_size: u32,
    pub feature_size: u32,
}
