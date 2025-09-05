use crate::tensor::desc::TensorDesc;

use super::{
    activations::{ActivationLayer, ActivationType},
    concat::ConcatLayer,
    conv2d::Conv2DLayer,
    element_wise::{ElementWiseLayer, ElementWiseOperation},
    input_buffer::InputLayer,
    layer::Layer,
    linear::LinearLayer,
    reshape::ReshapeLayer,
};

pub struct Layers;

impl Layers {
    pub fn input_buffer(out_features: i64) -> Box<dyn Layer> {
        Box::new(InputLayer::new(out_features))
    }

    pub fn input_buffer_with(out_features: i64, track_gradients: bool) -> Box<dyn Layer> {
        Box::new(InputLayer::new_with(out_features, track_gradients))
    }

    pub fn linear(in_features: i64, out_features: i64) -> Box<dyn Layer> {
        Box::new(LinearLayer::new(in_features, out_features))
    }

    pub fn linear_with(in_features: i64, out_features: i64, bias: bool) -> Box<dyn Layer> {
        Box::new(LinearLayer::new_with(in_features, out_features, bias))
    }

    pub fn conv2d(in_channels: i64, out_channels: i64) -> Box<dyn Layer> {
        Box::new(Conv2DLayer::new(in_channels, out_channels))
    }

    pub fn conv2d_with(
        in_features: i64,
        out_features: i64,
        kernel_h: i64,
        kernel_w: i64,
        stride_h: i64,
        stride_w: i64,
        padding_h: i64,
        padding_w: i64,
        bias: bool,
    ) -> Box<dyn Layer> {
        Box::new(Conv2DLayer::new_with(
            in_features,
            out_features,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            bias,
        ))
    }

    pub fn reshape(target_shape: TensorDesc) -> Box<dyn Layer> {
        Box::new(ReshapeLayer::new(target_shape))
    }

    pub fn flatten() -> Box<dyn Layer> {
        Box::new(ReshapeLayer::flatten())
    }

    pub fn concat(dim: usize) -> Box<dyn Layer> {
        Box::new(ConcatLayer::new(dim))
    }

    pub fn add() -> Box<dyn Layer> {
        Box::new(ElementWiseLayer::new(ElementWiseOperation::Add))
    }

    pub fn sub() -> Box<dyn Layer> {
        Box::new(ElementWiseLayer::new(ElementWiseOperation::Subtract))
    }

    pub fn mul() -> Box<dyn Layer> {
        Box::new(ElementWiseLayer::new(ElementWiseOperation::Multiply))
    }

    pub fn div() -> Box<dyn Layer> {
        Box::new(ElementWiseLayer::new(ElementWiseOperation::Divide))
    }

    pub fn max() -> Box<dyn Layer> {
        Box::new(ElementWiseLayer::new(ElementWiseOperation::Maximum))
    }

    pub fn min() -> Box<dyn Layer> {
        Box::new(ElementWiseLayer::new(ElementWiseOperation::Minimum))
    }

    pub fn relu() -> Box<dyn Layer> {
        Box::new(ActivationLayer::new(ActivationType::ReLU))
    }

    pub fn leakyrelu(alpha: f32) -> Box<dyn Layer> {
        Box::new(ActivationLayer::new(ActivationType::LeakyReLU(alpha)))
    }

    pub fn sigmoid() -> Box<dyn Layer> {
        Box::new(ActivationLayer::new(ActivationType::Sigmoid))
    }

    pub fn softmax(dim: usize) -> Box<dyn Layer> {
        Box::new(ActivationLayer::new(ActivationType::Softmax(dim)))
    }

    pub fn tanh() -> Box<dyn Layer> {
        Box::new(ActivationLayer::new(ActivationType::Tanh))
    }

    pub fn gelu() -> Box<dyn Layer> {
        Box::new(ActivationLayer::new(ActivationType::GELU))
    }

    pub fn silu() -> Box<dyn Layer> {
        Box::new(ActivationLayer::new(ActivationType::SiLU))
    }
}
