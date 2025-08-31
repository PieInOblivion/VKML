use super::r#trait::TensorStorageOps;
use crate::utils::expect_msg::ExpectMsg;
use std::ops::{Deref, DerefMut};
use std::sync::RwLock;

pub struct CpuTensorStorage {
    data: RwLock<Vec<u8>>,
}

impl CpuTensorStorage {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            data: RwLock::new(bytes),
        }
    }

}

impl TensorStorageOps for CpuTensorStorage {
    type ReadGuard<'a> = CpuReadGuard<'a>;
    type WriteGuard<'a> = CpuWriteGuard<'a>;

    fn read_data(&self) -> Self::ReadGuard<'_> {
        let guard = self.data.read().expect_msg("Failed to acquire read lock on CPU tensor data");
        CpuReadGuard { guard }
    }

    fn write_data(&self) -> Self::WriteGuard<'_> {
        let guard = self.data.write().expect_msg("Failed to acquire write lock on CPU tensor data");
        CpuWriteGuard { guard }
    }

    fn get_data(&self) -> Vec<u8> {
        self.data.read().expect_msg("Failed to acquire read lock on CPU tensor data").clone()
    }

    fn update_data(&self, data: Vec<u8>) {
        let mut guard = self.data.write().expect_msg("Failed to acquire write lock on CPU tensor data");
        if guard.len() != data.len() {
            panic!("Input data size mismatch: expected {} bytes, got {} bytes", guard.len(), data.len());
        }
        guard.copy_from_slice(&data);
    }

    fn size_in_bytes(&self) -> u64 {
        self.data.read().expect_msg("Failed to acquire read lock on CPU tensor data").len() as u64
    }

    fn is_allocated(&self) -> bool { true }

    fn gpu_idx(&self) -> Option<usize> { None }

    fn location_string(&self) -> String { "CPU Tensor".to_string() }
}

pub struct CpuReadGuard<'a> {
    guard: std::sync::RwLockReadGuard<'a, Vec<u8>>,
}

impl<'a> Deref for CpuReadGuard<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target { &self.guard }
}

pub struct CpuWriteGuard<'a> {
    guard: std::sync::RwLockWriteGuard<'a, Vec<u8>>,
}

impl<'a> Deref for CpuWriteGuard<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target { &self.guard }
}

impl<'a> DerefMut for CpuWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.guard }
}
