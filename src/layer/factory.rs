use crate::{tensor::TensorDesc, utils::OnnxAutoPad};

use super::{
    activations::{ActivationLayer, ActivationType},
    concat::ConcatLayer,
    conv::ConvLayer,
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

    pub fn conv_with(
        in_features: i64,
        out_features: i64,
        auto_pad: OnnxAutoPad,
        dilations: Vec<usize>,
        kernel_shape: Vec<usize>,
        pads: Vec<usize>,
        strides: Vec<usize>,
        bias: bool,
    ) -> Box<dyn Layer> {
        Box::new(ConvLayer::new_with(
            in_features,
            out_features,
            auto_pad,
            dilations,
            kernel_shape,
            pads,
            strides,
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

    pub fn sigmoid() -> Box<dyn Layer> {
        Box::new(ActivationLayer::new(ActivationType::Sigmoid))
    }

    pub fn softmax(axis: Option<i64>) -> Box<dyn Layer> {
        Box::new(ActivationLayer::new(ActivationType::Softmax(axis)))
    }
}
