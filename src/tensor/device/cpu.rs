use std::any::Any;

use crate::tensor::data::TensorData;

/// Simple CPU buffer: owns a Vec<u8>, no locks.
pub struct CpuData {
    pub data: Box<[u8]>,
}

impl CpuData {
    pub fn from_vec(v: Box<[u8]>) -> Self {
        Self { data: v }
    }
}

impl TensorData for CpuData {
    fn len_bytes(&self) -> usize {
        self.data.len()
    }

    fn read(&self) -> Box<[u8]> {
        self.data.clone()
    }

    fn write(&mut self, data: &[u8]) {
        assert_eq!(data.len(), self.data.len());
        self.data.copy_from_slice(data);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
