use crate::{
    gpu::gpu_memory::GPUMemory,
    tensor::{
        data::TensorData,
        desc::TensorDesc,
        device::{cpu::CpuData, gpu::GpuData, unallocated::UnallocatedData},
    },
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceId {
    CPU,
    GPU(usize),
    Unallocated,
}

pub struct Tensor {
    pub desc: TensorDesc,
    pub device: DeviceId,
    pub buffer: Arc<dyn TensorData>,
}

impl Tensor {
    /// Create a CPU-backed tensor from host data
    pub fn new_cpu(desc: TensorDesc, host_data: Vec<u8>) -> Self {
        let buf = CpuData::from_vec(host_data);
        Self {
            desc,
            device: DeviceId::CPU,
            buffer: Arc::new(buf),
        }
    }

    /// Create a GPU-backed tensor from an existing GPUMemory allocation
    pub fn new_gpu(desc: TensorDesc, gpu_idx: usize, memory: GPUMemory) -> Self {
        let buf = GpuData { memory };
        Self {
            desc,
            device: DeviceId::GPU(gpu_idx),
            buffer: Arc::new(buf),
        }
    }

    /// Create an unallocated tensor (placeholder) with no backing storage.
    /// Mostly used for planning.
    pub fn new_unallocated(desc: TensorDesc) -> Self {
        let buf = UnallocatedData::new();
        Self {
            desc,
            device: DeviceId::CPU,
            buffer: Arc::new(buf),
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self.device, DeviceId::CPU)
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self.device, DeviceId::GPU(_))
    }

    pub fn read(&self) -> Vec<u8> {
        self.buffer.read()
    }

    pub fn write(&mut self, data: &[u8]) {
        let mut_ref =
            Arc::get_mut(&mut self.buffer).expect("Tensor buffer not uniquely owned for write");
        mut_ref.write(data);
    }

    // The not super general functions below
    pub fn get_gpu_memory_or_panic(&self) -> &GPUMemory {
        // Try to downcast the trait object to GpuData
        let any = self.buffer.as_ref().as_any();
        let gpu = any
            .downcast_ref::<GpuData>()
            .expect("Tensor is not backed by GPU storage");
        &gpu.memory
    }
}
