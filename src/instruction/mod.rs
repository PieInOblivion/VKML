mod add;
pub use add::AddInstruction;
mod concat;
pub use concat::ConcatInstruction;
mod conv;
pub use conv::ConvInstruction;
mod div;
pub use div::DivInstruction;
mod expand;
pub use expand::ExpandInstruction;
mod gemm;
pub use gemm::GemmInstruction;
mod gpu_operations;
pub use gpu_operations::GPUOperation;
mod identity;
pub use identity::IdentityInstruction;
mod init_constant;
pub use init_constant::InitConstantInstruction;
mod init_he;
pub use init_he::InitHeInstruction;
mod init_uniform;
pub use init_uniform::InitUniformInstruction;
mod init_xavier;
pub use init_xavier::InitXavierInstruction;
mod instruction;
pub use instruction::Instruction;
mod matmul;
pub use matmul::MatMulInstruction;
mod max;
pub use max::MaxInstruction;
mod maxpool;
pub use maxpool::MaxPoolInstruction;
mod min;
pub use min::MinInstruction;
mod mul;
pub use mul::MulInstruction;
mod relu;
pub use relu::ReLUInstruction;
mod reducemean;
pub use reducemean::ReduceMeanInstruction;
mod reshape;
pub use reshape::ReshapeInstruction;
mod shape;
pub use shape::ShapeInstruction;
mod sigmoid;
pub use sigmoid::SigmoidInstruction;
mod softmax;
pub use softmax::SoftmaxInstruction;
mod sub;
pub use sub::SubInstruction;
mod transfer;
pub use transfer::TransferToDeviceInstruction;

use crate::{tensor::DeviceId, tensor_graph::TensorId, utils::OnnxAutoPad};

pub fn add(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(AddInstruction { src1, src2, dst })
}

pub fn concat(sources: Vec<TensorId>, dst: TensorId, dim: usize) -> Box<dyn Instruction> {
    Box::new(ConcatInstruction { sources, dst, dim })
}

pub fn conv(
    src: TensorId,
    weights: TensorId,
    bias: Option<TensorId>,
    dst: TensorId,
    auto_pad: OnnxAutoPad,
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
) -> Box<dyn Instruction> {
    Box::new(ConvInstruction {
        src,
        weights,
        bias,
        dst,
        auto_pad,
        dilations,
        group,
        kernel_shape,
        pads,
        strides,
    })
}

pub fn div(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(DivInstruction { src1, src2, dst })
}

pub fn expand(src: TensorId, dst: TensorId, shape: Vec<i64>) -> Box<dyn Instruction> {
    Box::new(ExpandInstruction {
        src,
        dst,
        shape_values: shape,
    })
}

pub fn identity(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(IdentityInstruction { src, dst })
}

pub fn init_constant(dst: TensorId, constant: Vec<u8>) -> Box<dyn Instruction> {
    Box::new(InitConstantInstruction { dst, constant })
}

pub fn init_he(dst: TensorId) -> Box<dyn Instruction> {
    Box::new(InitHeInstruction { dst })
}

pub fn init_uniform(dst: TensorId, min: f32, max: f32) -> Box<dyn Instruction> {
    Box::new(InitUniformInstruction { dst, min, max })
}

pub fn init_xavier(dst: TensorId) -> Box<dyn Instruction> {
    Box::new(InitXavierInstruction { dst })
}

pub fn matmul(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(MatMulInstruction { src1, src2, dst })
}

pub fn max(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(MaxInstruction { src1, src2, dst })
}

pub fn maxpool(
    src: TensorId,
    dst: TensorId,
    auto_pad: OnnxAutoPad,
    dilations: Vec<i64>,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
    ceil_mode: bool,
) -> Box<dyn Instruction> {
    Box::new(MaxPoolInstruction {
        src,
        dst,
        auto_pad,
        dilations,
        kernel_shape,
        pads,
        strides,
        ceil_mode,
    })
}

pub fn min(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(MinInstruction { src1, src2, dst })
}

pub fn mul(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(MulInstruction { src1, src2, dst })
}

pub fn relu(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(ReLUInstruction { src, dst })
}

pub fn reducemean(
    src: TensorId,
    axes: Option<Vec<i64>>,
    keepdims: i64,
    noop_with_empty_axes: i64,
    dst: TensorId,
) -> Box<dyn Instruction> {
    Box::new(ReduceMeanInstruction {
        src,
        axes,
        keepdims,
        noop_with_empty_axes,
        dst,
    })
}

pub fn reshape(
    src: TensorId,
    dst: TensorId,
    new_shape: Vec<i64>,
    allowzero: Option<i64>,
) -> Box<dyn Instruction> {
    Box::new(ReshapeInstruction {
        src,
        dst,
        shape_values: new_shape,
        allowzero,
    })
}

pub fn shape(
    src: TensorId,
    dst: TensorId,
    start: Option<i64>,
    end: Option<i64>,
) -> Box<dyn Instruction> {
    Box::new(ShapeInstruction {
        src,
        dst,
        start,
        end,
    })
}

pub fn sigmoid(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(SigmoidInstruction { src, dst })
}

pub fn softmax(src: TensorId, dst: TensorId, axis: Option<i64>) -> Box<dyn Instruction> {
    Box::new(SoftmaxInstruction { src, dst, axis })
}

pub fn sub(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(SubInstruction { src1, src2, dst })
}

pub fn transfer(
    src: TensorId,
    dst: TensorId,
    source_device: DeviceId,
    target_device: DeviceId,
) -> Box<dyn Instruction> {
    Box::new(TransferToDeviceInstruction {
        src,
        dst,
        source_device,
        target_device,
    })
}

pub fn gemm(
    a: TensorId,
    b: TensorId,
    c: Option<TensorId>,
    y: TensorId,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) -> Box<dyn Instruction> {
    Box::new(GemmInstruction {
        a,
        b,
        c,
        y,
        alpha,
        beta,
        trans_a,
        trans_b,
    })
}
