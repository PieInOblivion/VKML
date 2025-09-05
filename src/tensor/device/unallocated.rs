use std::any::Any;

use crate::tensor::data::TensorData;

/// Unallocated buffer placeholder: no backing storage yet.
pub struct UnallocatedData {}

impl UnallocatedData {
    pub fn new() -> Self {
        Self {}
    }
}

impl TensorData for UnallocatedData {
    fn len_bytes(&self) -> usize {
        panic!("Unallocated buffer has no size; consult TensorDesc")
    }

    fn read(&self) -> Vec<u8> {
        panic!("Attempted to read from unallocated buffer")
    }

    fn write(&mut self, _data: &[u8]) {
        panic!("Attempted to write to unallocated buffer")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
