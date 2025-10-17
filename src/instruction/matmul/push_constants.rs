#[repr(C)]
pub struct MatMul1D2DPushConstants {
    pub k: u32,
    pub n: u32,
    pub stride_a: u32,
    pub stride_b0: u32,
    pub stride_b1: u32,
    pub stride_c: u32,
}

#[repr(C)]
pub struct MatMul2D1DPushConstants {
    pub m: u32,
    pub k: u32,
    pub stride_a0: u32,
    pub stride_a1: u32,
    pub stride_b: u32,
    pub stride_c: u32,
}

#[repr(C)]
pub struct MatMul2D2DPushConstants {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub stride_a0: u32,
    pub stride_a1: u32,
    pub stride_b0: u32,
    pub stride_b1: u32,
    pub stride_c0: u32,
    pub stride_c1: u32,
}

#[repr(C)]
pub struct MatMul2D3DPushConstants {
    pub batch: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub stride_a0: u32,
    pub stride_a1: u32,
    pub stride_b0: u32,
    pub stride_b1: u32,
    pub stride_b2: u32,
    pub stride_c0: u32,
    pub stride_c1: u32,
    pub stride_c2: u32,
}

#[repr(C)]
pub struct MatMul3D2DPushConstants {
    pub batch: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub stride_a0: u32,
    pub stride_a1: u32,
    pub stride_a2: u32,
    pub stride_b0: u32,
    pub stride_b1: u32,
    pub stride_c0: u32,
    pub stride_c1: u32,
    pub stride_c2: u32,
}

#[repr(C)]
pub struct MatMul3D3DPushConstants {
    pub batch: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub stride_a0: u32,
    pub stride_a1: u32,
    pub stride_a2: u32,
    pub stride_b0: u32,
    pub stride_b1: u32,
    pub stride_b2: u32,
    pub stride_c0: u32,
    pub stride_c1: u32,
    pub stride_c2: u32,
}

#[repr(C)]
pub struct MatMul3D1DPushConstants {
    pub batch: u32,
    pub m: u32,
    pub k: u32,
    pub stride_a0: u32,
    pub stride_a1: u32,
    pub stride_a2: u32,
    pub stride_b: u32,
    pub stride_c0: u32,
    pub stride_c1: u32,
}

#[repr(C)]
pub struct MatMul1D3DPushConstants {
    pub batch: u32,
    pub k: u32,
    pub n: u32,
    pub stride_a: u32,
    pub stride_b0: u32,
    pub stride_b1: u32,
    pub stride_b2: u32,
    pub stride_c0: u32,
    pub stride_c1: u32,
}
