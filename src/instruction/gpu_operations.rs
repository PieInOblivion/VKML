use std::sync::OnceLock;

// Lazy-loaded shader storage
// Array is indexed by GPUOperation discriminant
// using __Count sentinal might not be the most reliable
// std::mem::variant_count is currently unstable
static SHADERS: [OnceLock<Vec<u8>>; GPUOperation::__Count as usize] =
    [const { OnceLock::new() }; GPUOperation::__Count as usize];

#[allow(non_camel_case_types)]
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum GPUOperation {
    Addition_F32_F32_F32,
    Addition_F16_F16_F16,
    Subtract_F32_F32_F32,
    Multiply_F32_F32_F32,
    Divide_F32_F32_F32,
    Expand_F32_F32,
    Expand_F16_F16,
    Maximum_F32_F32_F32,
    Minimum_F32_F32_F32,
    ReLU_F32_F32,
    ReLU_F16_F16,
    Sigmoid_F32_F32,
    Softmax_F32_F32,
    Softmax_F16_F16,
    Conv1D_F32_F32_F32_F32,
    Conv2D_F32_F32_F32_F32,
    Conv3D_F32_F32_F32_F32,
    Conv2D_F16_F16_F16_F16,
    MaxPool1D_F32_F32,
    MaxPool2D_F32_F32,
    MaxPool3D_F32_F32,
    MaxPool2D_F16_F16,
    MatMul1D2D_F32_F32_F32,
    MatMul2D1D_F32_F32_F32,
    MatMul2D2D_F32_F32_F32,
    MatMul2D3D_F32_F32_F32,
    MatMul3D2D_F32_F32_F32,
    MatMul3D3D_F32_F32_F32,
    MatMul3D1D_F32_F32_F32,
    MatMul1D3D_F32_F32_F32,
    MatMul2D2D_F16_F16_F16,
    MatMul2D2D_F16_F16_F16_Coop_16_16_16_SG_32,
    InitXavier_F32,
    InitHe_F32,
    InitUniform_F32,
    InitConstant,
    Shape_Write_I64,
    ReduceMean_F32_F32,
    ReduceMean_F16_F16,
    Gemm_F32_F32_F32_F32,
    Gemm_F16_F16_F16_F16,
    __Count,
}

impl GPUOperation {
    pub fn get_shader_bytes(&self) -> &[u8] {
        let idx = *self as usize;
        SHADERS[idx].get_or_init(|| {
            let filename = self.shader_filename();
            let path = format!("{}/shaders/{}", env!("OUT_DIR"), filename);
            std::fs::read(&path)
                .unwrap_or_else(|e| panic!("Failed to load shader file '{}': {}", path, e))
        })
    }

    pub fn is_init(&self) -> bool {
        matches!(
            self,
            GPUOperation::InitXavier_F32
                | GPUOperation::InitHe_F32
                | GPUOperation::InitUniform_F32
                | GPUOperation::InitConstant
        )
    }

    fn shader_filename(&self) -> &'static str {
        match self {
            GPUOperation::Addition_F32_F32_F32 => "f32_f32_f32_add.spv",
            GPUOperation::Addition_F16_F16_F16 => "f16_f16_f16_add.spv",
            GPUOperation::Subtract_F32_F32_F32 => "f32_f32_f32_sub.spv",
            GPUOperation::Multiply_F32_F32_F32 => "f32_f32_f32_mul.spv",
            GPUOperation::Divide_F32_F32_F32 => "f32_f32_f32_div.spv",
            GPUOperation::Expand_F32_F32 => "f32_f32_expand.spv",
            GPUOperation::Expand_F16_F16 => "f16_f16_expand.spv",
            GPUOperation::Maximum_F32_F32_F32 => "f32_f32_f32_max.spv",
            GPUOperation::Minimum_F32_F32_F32 => "f32_f32_f32_min.spv",

            GPUOperation::ReLU_F32_F32 => "f32_f32_relu.spv",
            GPUOperation::ReLU_F16_F16 => "f16_f16_relu.spv",
            GPUOperation::Sigmoid_F32_F32 => "f32_f32_sigmoid.spv",
            GPUOperation::Softmax_F32_F32 => "f32_f32_softmax.spv",
            GPUOperation::Softmax_F16_F16 => "f16_f16_softmax.spv",

            GPUOperation::Conv1D_F32_F32_F32_F32 => "f32_f32_f32_f32_conv1d.spv",
            GPUOperation::Conv2D_F32_F32_F32_F32 => "f32_f32_f32_f32_conv2d.spv",
            GPUOperation::Conv3D_F32_F32_F32_F32 => "f32_f32_f32_f32_conv3d.spv",
            GPUOperation::Conv2D_F16_F16_F16_F16 => "f16_f16_f16_f16_conv2d.spv",

            GPUOperation::MaxPool1D_F32_F32 => "f32_f32_maxpool1d.spv",
            GPUOperation::MaxPool2D_F32_F32 => "f32_f32_maxpool2d.spv",
            GPUOperation::MaxPool3D_F32_F32 => "f32_f32_maxpool3d.spv",
            GPUOperation::MaxPool2D_F16_F16 => "f16_f16_maxpool2d.spv",

            GPUOperation::MatMul1D2D_F32_F32_F32 => "f32_f32_f32_matmul_1d_2d.spv",
            GPUOperation::MatMul2D1D_F32_F32_F32 => "f32_f32_f32_matmul_2d_1d.spv",
            GPUOperation::MatMul2D2D_F32_F32_F32 => "f32_f32_f32_matmul_2d_2d.spv",
            GPUOperation::MatMul2D3D_F32_F32_F32 => "f32_f32_f32_matmul_2d_3d.spv",
            GPUOperation::MatMul3D2D_F32_F32_F32 => "f32_f32_f32_matmul_3d_2d.spv",
            GPUOperation::MatMul3D3D_F32_F32_F32 => "f32_f32_f32_matmul_3d_3d.spv",
            GPUOperation::MatMul3D1D_F32_F32_F32 => "f32_f32_f32_matmul_3d_1d.spv",
            GPUOperation::MatMul1D3D_F32_F32_F32 => "f32_f32_f32_matmul_1d_3d.spv",
            GPUOperation::MatMul2D2D_F16_F16_F16 => "f16_f16_f16_matmul_2d_2d.spv",
            GPUOperation::MatMul2D2D_F16_F16_F16_Coop_16_16_16_SG_32 => {
                "f16_f16_f16_matmul_2d_2d_coop_16_16_16_sg_32.spv"
            }

            GPUOperation::InitXavier_F32 => "f32_init_xavier.spv",
            GPUOperation::InitHe_F32 => "f32_init_he.spv",
            GPUOperation::InitUniform_F32 => "f32_init_uniform.spv",
            GPUOperation::InitConstant => "init_constant.spv",

            GPUOperation::Shape_Write_I64 => "i64_shape.spv",
            GPUOperation::ReduceMean_F32_F32 => "f32_f32_reducemean.spv",
            GPUOperation::ReduceMean_F16_F16 => "f16_f16_reducemean.spv",
            GPUOperation::Gemm_F32_F32_F32_F32 => "f32_f32_f32_f32_gemm.spv",
            GPUOperation::Gemm_F16_F16_F16_F16 => "f16_f16_f16_f16_gemm.spv",
            GPUOperation::__Count => unreachable!("__Count is not a valid shader operation"),
        }
    }
}
