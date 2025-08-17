use crate::tensor::storage::r#trait::TensorStorage;

use super::tensor_desc::TensorDesc;

pub struct ComputeTensor {
    pub desc: TensorDesc,
    pub data: Box<dyn TensorStorage>,
}
