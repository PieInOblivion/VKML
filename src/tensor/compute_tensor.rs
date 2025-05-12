use super::{tensor_data::TensorData, tensor_desc::TensorDesc};

pub struct ComputeTensor {
    pub desc: TensorDesc,
    pub data: TensorData,
}
