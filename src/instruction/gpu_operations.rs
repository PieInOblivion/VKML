macro_rules! include_shader {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $name))
    };
}

// shader byte constants centralised here so operations can map directly to their SPIR-V
// The actual .spv files are generated into OUT_DIR by the build script and resolved
// via the include_shader! macro
const ADD_SHADER_F32_F32_F32: &[u8] = include_shader!("f32_f32_f32_add.spv");
const ADD_SHADER_F16_F16_F16: &[u8] = include_shader!("f16_f16_f16_add.spv");
const SUB_SHADER_F32_F32_F32: &[u8] = include_shader!("f32_f32_f32_sub.spv");
const MUL_SHADER_F32_F32_F32: &[u8] = include_shader!("f32_f32_f32_mul.spv");
const DIV_SHADER_F32_F32_F32: &[u8] = include_shader!("f32_f32_f32_div.spv");
const MAX_SHADER_F32_F32_F32: &[u8] = include_shader!("f32_f32_f32_max.spv");
const MIN_SHADER_F32_F32_F32: &[u8] = include_shader!("f32_f32_f32_min.spv");

const RELU_SHADER_F32_F32: &[u8] = include_shader!("f32_f32_relu.spv");
const RELU_SHADER_F16_F16: &[u8] = include_shader!("f16_f16_relu.spv");
const SIGMOID_SHADER_F32_F32: &[u8] = include_shader!("f32_f32_sigmoid.spv");
const SOFTMAX_SHADER_F32_F32: &[u8] = include_shader!("f32_f32_softmax.spv");
const SOFTMAX_SHADER_F16_F16: &[u8] = include_shader!("f16_f16_softmax.spv");

const CONV1D_SHADER_F32_F32_F32_F32: &[u8] = include_shader!("f32_f32_f32_f32_conv1d.spv");
const CONV2D_SHADER_F32_F32_F32_F32: &[u8] = include_shader!("f32_f32_f32_f32_conv2d.spv");
const CONV2D_SHADER_F16_F16_F16_F16: &[u8] = include_shader!("f16_f16_f16_f16_conv2d.spv");
const CONV3D_SHADER_F32_F32_F32_F32: &[u8] = include_shader!("f32_f32_f32_f32_conv3d.spv");

const MAXPOOL1D_SHADER_F32_F32: &[u8] = include_shader!("f32_f32_maxpool1d.spv");
const MAXPOOL2D_SHADER_F32_F32: &[u8] = include_shader!("f32_f32_maxpool2d.spv");
const MAXPOOL2D_SHADER_F16_F16: &[u8] = include_shader!("f16_f16_maxpool2d.spv");
const MAXPOOL3D_SHADER_F32_F32: &[u8] = include_shader!("f32_f32_maxpool3d.spv");

const MATMUL_1D_2D_SHADER_F32: &[u8] = include_shader!("f32_f32_f32_matmul_1d_2d.spv");
const MATMUL_2D_1D_SHADER_F32: &[u8] = include_shader!("f32_f32_f32_matmul_2d_1d.spv");
const MATMUL_2D_2D_SHADER_F32: &[u8] = include_shader!("f32_f32_f32_matmul_2d_2d.spv");
const MATMUL_2D_2D_SHADER_F16: &[u8] = include_shader!("f16_f16_f16_matmul_2d_2d.spv");
const MATMUL_2D_2D_SHADER_F16_COOP_16_16_16_SG_32: &[u8] =
    include_shader!("f16_f16_f16_matmul_2d_2d_coop_16_16_16_sg_32.spv");
const MATMUL_2D_3D_SHADER_F32: &[u8] = include_shader!("f32_f32_f32_matmul_2d_3d.spv");
const MATMUL_3D_2D_SHADER_F32: &[u8] = include_shader!("f32_f32_f32_matmul_3d_2d.spv");
const MATMUL_3D_3D_SHADER_F32: &[u8] = include_shader!("f32_f32_f32_matmul_3d_3d.spv");
const MATMUL_3D_1D_SHADER_F32: &[u8] = include_shader!("f32_f32_f32_matmul_3d_1d.spv");
const MATMUL_1D_3D_SHADER_F32: &[u8] = include_shader!("f32_f32_f32_matmul_1d_3d.spv");

const INIT_XAVIER_SHADER_F32: &[u8] = include_shader!("f32_init_xavier.spv");
const INIT_HE_SHADER_F32: &[u8] = include_shader!("f32_init_he.spv");
const INIT_UNIFORM_SHADER_F32: &[u8] = include_shader!("f32_init_uniform.spv");
const INIT_CONSTANT_SHADER: &[u8] = include_shader!("init_constant.spv");

const SHAPE_SHADER_I64: &[u8] = include_shader!("i64_shape.spv");
const REDUCE_MEAN_SHADER_F32: &[u8] = include_shader!("f32_reducemean_mean.spv");

const GEMM_SHADER_F32_F32_F32_F32: &[u8] = include_shader!("f32_f32_f32_f32_gemm.spv");
const GEMM_SHADER_F16_F16_F16_F16: &[u8] = include_shader!("f16_f16_f16_f16_gemm.spv");

#[allow(non_camel_case_types)]
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum GPUOperation {
    Addition_F32_F32_F32,
    Addition_F16_F16_F16,
    Subtract_F32_F32_F32,
    Multiply_F32_F32_F32,
    Divide_F32_F32_F32,
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
    ReduceMean_F32,
    Gemm_F32_F32_F32_F32,
    Gemm_F16_F16_F16_F16,
}

impl GPUOperation {
    // NOTE: std::mem::variant_count is currently unstable
    //pub const VARIANT_COUNT: usize = 29;

    pub fn get_shader_bytes(&self) -> &[u8] {
        match self {
            GPUOperation::Addition_F32_F32_F32 => ADD_SHADER_F32_F32_F32,
            GPUOperation::Addition_F16_F16_F16 => ADD_SHADER_F16_F16_F16,
            GPUOperation::Subtract_F32_F32_F32 => SUB_SHADER_F32_F32_F32,
            GPUOperation::Multiply_F32_F32_F32 => MUL_SHADER_F32_F32_F32,
            GPUOperation::Divide_F32_F32_F32 => DIV_SHADER_F32_F32_F32,
            GPUOperation::Maximum_F32_F32_F32 => MAX_SHADER_F32_F32_F32,
            GPUOperation::Minimum_F32_F32_F32 => MIN_SHADER_F32_F32_F32,
            GPUOperation::ReLU_F32_F32 => RELU_SHADER_F32_F32,
            GPUOperation::ReLU_F16_F16 => RELU_SHADER_F16_F16,
            GPUOperation::Sigmoid_F32_F32 => SIGMOID_SHADER_F32_F32,
            GPUOperation::Softmax_F32_F32 => SOFTMAX_SHADER_F32_F32,
            GPUOperation::Softmax_F16_F16 => SOFTMAX_SHADER_F16_F16,
            GPUOperation::Conv1D_F32_F32_F32_F32 => CONV1D_SHADER_F32_F32_F32_F32,
            GPUOperation::Conv2D_F32_F32_F32_F32 => CONV2D_SHADER_F32_F32_F32_F32,
            GPUOperation::Conv3D_F32_F32_F32_F32 => CONV3D_SHADER_F32_F32_F32_F32,
            GPUOperation::Conv2D_F16_F16_F16_F16 => CONV2D_SHADER_F16_F16_F16_F16,
            GPUOperation::MaxPool1D_F32_F32 => MAXPOOL1D_SHADER_F32_F32,
            GPUOperation::MaxPool2D_F32_F32 => MAXPOOL2D_SHADER_F32_F32,
            GPUOperation::MaxPool3D_F32_F32 => MAXPOOL3D_SHADER_F32_F32,
            GPUOperation::MaxPool2D_F16_F16 => MAXPOOL2D_SHADER_F16_F16,
            GPUOperation::MatMul1D2D_F32_F32_F32 => MATMUL_1D_2D_SHADER_F32,
            GPUOperation::MatMul2D1D_F32_F32_F32 => MATMUL_2D_1D_SHADER_F32,
            GPUOperation::MatMul2D2D_F32_F32_F32 => MATMUL_2D_2D_SHADER_F32,
            GPUOperation::MatMul2D3D_F32_F32_F32 => MATMUL_2D_3D_SHADER_F32,
            GPUOperation::MatMul3D2D_F32_F32_F32 => MATMUL_3D_2D_SHADER_F32,
            GPUOperation::MatMul3D3D_F32_F32_F32 => MATMUL_3D_3D_SHADER_F32,
            GPUOperation::MatMul3D1D_F32_F32_F32 => MATMUL_3D_1D_SHADER_F32,
            GPUOperation::MatMul1D3D_F32_F32_F32 => MATMUL_1D_3D_SHADER_F32,
            GPUOperation::MatMul2D2D_F16_F16_F16 => MATMUL_2D_2D_SHADER_F16,
            GPUOperation::MatMul2D2D_F16_F16_F16_Coop_16_16_16_SG_32 => {
                MATMUL_2D_2D_SHADER_F16_COOP_16_16_16_SG_32
            }
            GPUOperation::InitXavier_F32 => INIT_XAVIER_SHADER_F32,
            GPUOperation::InitHe_F32 => INIT_HE_SHADER_F32,
            GPUOperation::InitUniform_F32 => INIT_UNIFORM_SHADER_F32,
            GPUOperation::InitConstant => INIT_CONSTANT_SHADER,
            GPUOperation::Shape_Write_I64 => SHAPE_SHADER_I64,
            GPUOperation::ReduceMean_F32 => REDUCE_MEAN_SHADER_F32,
            GPUOperation::Gemm_F32_F32_F32_F32 => GEMM_SHADER_F32_F32_F32_F32,
            GPUOperation::Gemm_F16_F16_F16_F16 => GEMM_SHADER_F16_F16_F16_F16,
        }
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

    pub fn expect_init(&self) {
        if !self.is_init() {
            panic!("Expected init GPUOperation, found {:?}", self);
        }
    }
}
