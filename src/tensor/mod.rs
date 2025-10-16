mod cell;
pub use cell::TensorCell;
mod data;
mod desc;
pub use desc::TensorDesc;
mod device;
mod tensor;
pub use tensor::{DeviceId, Tensor};
