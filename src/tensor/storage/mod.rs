mod cpu;
mod gpu;
mod r#trait;
mod unallocated;

pub use cpu::{CpuReadGuard, CpuTensorStorage, CpuWriteGuard};
pub use gpu::{GpuReadGuard, GpuTensorStorage, GpuWriteGuard};
pub use r#trait::TensorStorageOps;
pub use unallocated::{UnallocatedGuard, UnallocatedTensorStorage};

use std::ops::{Deref, DerefMut};

/// Main enum for tensor storage
pub enum TensorStorage {
    CPU(CpuTensorStorage),
    GPU(GpuTensorStorage),
    Unallocated(UnallocatedTensorStorage),
}

/// Union type for read guards
pub enum ReadGuard<'a> {
    CPU(CpuReadGuard<'a>),
    GPU(GpuReadGuard),
    Unallocated(UnallocatedGuard),
}

/// Union type for write guards  
pub enum WriteGuard<'a> {
    CPU(CpuWriteGuard<'a>),
    GPU(GpuWriteGuard<'a>),
    Unallocated(UnallocatedGuard),
}

impl<'a> Deref for ReadGuard<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            ReadGuard::CPU(guard) => guard.deref(),
            ReadGuard::GPU(guard) => guard.deref(),
            ReadGuard::Unallocated(guard) => guard.deref(),
        }
    }
}

impl<'a> Deref for WriteGuard<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            WriteGuard::CPU(guard) => guard.deref(),
            WriteGuard::GPU(guard) => guard.deref(),
            WriteGuard::Unallocated(guard) => guard.deref(),
        }
    }
}

impl<'a> DerefMut for WriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            WriteGuard::CPU(guard) => guard.deref_mut(),
            WriteGuard::GPU(guard) => guard.deref_mut(),
            WriteGuard::Unallocated(guard) => guard.deref_mut(),
        }
    }
}

impl TensorStorage {
    pub fn new_cpu(bytes: Vec<u8>) -> Self {
        TensorStorage::CPU(CpuTensorStorage::new(bytes))
    }

    pub fn new_gpu(gpu_idx: usize, memory: crate::gpu::gpu_memory::GPUMemory) -> Self {
        TensorStorage::GPU(GpuTensorStorage::new(gpu_idx, memory))
    }

    pub fn new_unallocated() -> Self {
        TensorStorage::Unallocated(UnallocatedTensorStorage)
    }

    /// Get read-only access to tensor data
    pub fn read_data(&self) -> ReadGuard<'_> {
        match self {
            TensorStorage::CPU(storage) => ReadGuard::CPU(storage.read_data()),
            TensorStorage::GPU(storage) => ReadGuard::GPU(storage.read_data()),
            TensorStorage::Unallocated(storage) => ReadGuard::Unallocated(storage.read_data()),
        }
    }

    /// Get mutable access to tensor data
    pub fn write_data(&self) -> WriteGuard<'_> {
        match self {
            TensorStorage::CPU(storage) => WriteGuard::CPU(storage.write_data()),
            TensorStorage::GPU(storage) => WriteGuard::GPU(storage.write_data()),
            TensorStorage::Unallocated(storage) => WriteGuard::Unallocated(storage.write_data()),
        }
    }

    /// Read all data from storage as raw bytes (full copy)
    pub fn get_data(&self) -> Vec<u8> {
        match self {
            TensorStorage::CPU(storage) => storage.get_data(),
            TensorStorage::GPU(storage) => storage.get_data(),
            TensorStorage::Unallocated(storage) => storage.get_data(),
        }
    }

    /// Update storage with new raw bytes - panics on size mismatch or failure
    pub fn update_data(&self, data: Vec<u8>) {
        match self {
            TensorStorage::CPU(storage) => storage.update_data(data),
            TensorStorage::GPU(storage) => storage.update_data(data),
            TensorStorage::Unallocated(storage) => storage.update_data(data),
        }
    }

    /// Get size in bytes
    pub fn size_in_bytes(&self) -> u64 {
        match self {
            TensorStorage::CPU(storage) => storage.size_in_bytes(),
            TensorStorage::GPU(storage) => storage.size_in_bytes(),
            TensorStorage::Unallocated(storage) => storage.size_in_bytes(),
        }
    }

    /// Check if storage is allocated
    pub fn is_allocated(&self) -> bool {
        match self {
            TensorStorage::CPU(storage) => storage.is_allocated(),
            TensorStorage::GPU(storage) => storage.is_allocated(),
            TensorStorage::Unallocated(storage) => storage.is_allocated(),
        }
    }

    /// Get GPU index if this is GPU storage, None for CPU/other storage
    pub fn gpu_idx(&self) -> Option<usize> {
        match self {
            TensorStorage::CPU(storage) => storage.gpu_idx(),
            TensorStorage::GPU(storage) => storage.gpu_idx(),
            TensorStorage::Unallocated(storage) => storage.gpu_idx(),
        }
    }

    /// Get human-readable location description
    pub fn location_string(&self) -> String {
        match self {
            TensorStorage::CPU(storage) => storage.location_string(),
            TensorStorage::GPU(storage) => storage.location_string(),
            TensorStorage::Unallocated(storage) => storage.location_string(),
        }
    }
}
