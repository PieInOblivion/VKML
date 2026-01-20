use crate::{gpu::gpu_memory::GPUMemory, tensor::desc::TensorDesc};
use std::borrow::Cow;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum DeviceId {
    Cpu,
    Gpu(usize),
}

enum TensorStorage {
    Cpu(Box<[u8]>),
    Gpu { gpu_idx: usize, memory: GPUMemory },
}

pub struct Tensor {
    desc: TensorDesc,
    storage: TensorStorage,
}

impl Tensor {
    /// Create a CPU-backed tensor from host data
    pub fn new_cpu(desc: TensorDesc, host_data: Box<[u8]>) -> Self {
        Self {
            desc,
            storage: TensorStorage::Cpu(host_data),
        }
    }

    /// Create a GPU-backed tensor from an existing GPUMemory allocation
    pub fn new_gpu(desc: TensorDesc, gpu_idx: usize, memory: GPUMemory) -> Self {
        Self {
            desc,
            storage: TensorStorage::Gpu { gpu_idx, memory },
        }
    }

    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    pub fn desc_mut(&mut self) -> &mut TensorDesc {
        &mut self.desc
    }

    pub fn device(&self) -> DeviceId {
        match &self.storage {
            TensorStorage::Cpu(_) => DeviceId::Cpu,
            TensorStorage::Gpu { gpu_idx, .. } => DeviceId::Gpu(*gpu_idx),
        }
    }

    /// Return length in bytes of the underlying storage.
    pub fn len_bytes(&self) -> usize {
        match &self.storage {
            TensorStorage::Cpu(data) => data.len(),
            TensorStorage::Gpu { memory, .. } => memory.size as usize,
        }
    }

    /// Read bytes, borrowing CPU storage when possible.
    ///
    /// - CPU tensors return `Cow::Borrowed(&[u8])` (no allocation)
    /// - GPU tensors return `Cow::Owned(Vec<u8>)` (requires a copy back to host)
    pub fn read(&self) -> Cow<'_, [u8]> {
        match &self.storage {
            TensorStorage::Cpu(data) => Cow::Borrowed(data),
            TensorStorage::Gpu { memory, .. } => Cow::Owned(
                memory
                    .read_memory()
                    .expect("Failed to read GPU memory")
                    .into_vec(),
            ),
        }
    }

    pub fn write(&mut self, data: &[u8]) {
        match &mut self.storage {
            TensorStorage::Cpu(buf) => {
                assert_eq!(data.len(), buf.len());
                buf.copy_from_slice(data);
            }
            TensorStorage::Gpu { memory, .. } => {
                assert_eq!(data.len(), memory.size as usize);
                memory
                    .copy_into(data)
                    .expect("Failed to copy data into GPU memory");
            }
        }
    }

    // The not super general functions below
    pub fn get_gpu_memory_or_panic(&self) -> &GPUMemory {
        match &self.storage {
            TensorStorage::Gpu { memory, .. } => memory,
            TensorStorage::Cpu(_) => panic!("Tensor is not backed by GPU storage"),
        }
    }

    pub fn get_cpu_memory_slice_or_panic(&self) -> &[u8] {
        match &self.storage {
            TensorStorage::Cpu(data) => data,
            TensorStorage::Gpu { .. } => panic!("Tensor is not backed by CPU storage"),
        }
    }

    pub fn get_cpu_memory_mut_slice_or_panic(&mut self) -> &mut [u8] {
        match &mut self.storage {
            TensorStorage::Cpu(data) => data,
            TensorStorage::Gpu { .. } => panic!("Tensor is not backed by CPU storage"),
        }
    }
}
