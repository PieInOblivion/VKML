use super::r#trait::TensorStorageOps;
use crate::utils::expect_msg::ExpectMsg;
use std::ops::{Deref, DerefMut};
use std::sync::RwLock;

pub struct CpuTensorStorage {
    data: RwLock<Vec<f32>>,
}

impl CpuTensorStorage {
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data: RwLock::new(data),
        }
    }

    pub fn with_zeros(size: usize) -> Self {
        Self {
            data: RwLock::new(vec![0.0; size]),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: RwLock::new(Vec::with_capacity(capacity)),
        }
    }
}

impl TensorStorageOps for CpuTensorStorage {
    type ReadGuard<'a> = CpuReadGuard<'a>;
    type WriteGuard<'a> = CpuWriteGuard<'a>;

    fn read_data(&self) -> Self::ReadGuard<'_> {
        CpuReadGuard(
            self.data
                .read()
                .expect_msg("Failed to acquire read lock on CPU tensor data"),
        )
    }

    fn write_data(&self) -> Self::WriteGuard<'_> {
        CpuWriteGuard(
            self.data
                .write()
                .expect_msg("Failed to acquire write lock on CPU tensor data"),
        )
    }

    fn get_data(&self) -> Vec<f32> {
        self.data
            .read()
            .expect_msg("Failed to acquire read lock on CPU tensor data")
            .clone()
    }

    fn update_data(&self, data: Vec<f32>) {
        let mut guard = self
            .data
            .write()
            .expect_msg("Failed to acquire write lock on CPU tensor data");

        let expected_elements = guard.len();
        if data.len() != expected_elements {
            panic!(
                "Input data size mismatch: expected {} elements, got {}",
                expected_elements,
                data.len()
            );
        }
        *guard = data;
    }

    fn size_in_bytes(&self) -> u64 {
        let guard = self
            .data
            .read()
            .expect_msg("Failed to acquire read lock on CPU tensor data");
        (guard.len() * std::mem::size_of::<f32>()) as u64
    }

    fn is_allocated(&self) -> bool {
        true
    }

    fn gpu_idx(&self) -> Option<usize> {
        None
    }

    fn location_string(&self) -> String {
        "CPU Tensor".to_string()
    }
}

// Wrapper for RwLockReadGuard to implement Deref for [f32]
pub struct CpuReadGuard<'a>(std::sync::RwLockReadGuard<'a, Vec<f32>>);

impl<'a> Deref for CpuReadGuard<'a> {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Wrapper for RwLockWriteGuard to implement DerefMut for [f32]
pub struct CpuWriteGuard<'a>(std::sync::RwLockWriteGuard<'a, Vec<f32>>);

impl<'a> Deref for CpuWriteGuard<'a> {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> DerefMut for CpuWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
