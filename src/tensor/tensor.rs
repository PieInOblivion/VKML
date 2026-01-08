use crate::{
    gpu::gpu_memory::GPUMemory,
    tensor::{
        data::TensorData,
        desc::TensorDesc,
        device::{cpu::CpuData, gpu::GpuData},
    },
};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum DeviceId {
    Cpu,
    Gpu(usize),
}

pub struct Tensor {
    desc: TensorDesc,
    device: DeviceId,
    buffer: Box<dyn TensorData>,
}

impl Tensor {
    /// Create a CPU-backed tensor from host data
    pub fn new_cpu(desc: TensorDesc, host_data: Box<[u8]>) -> Self {
        let buf = CpuData::from_boxed_slice(host_data);
        Self {
            desc,
            device: DeviceId::Cpu,
            buffer: Box::new(buf),
        }
    }

    /// Create a GPU-backed tensor from an existing GPUMemory allocation
    pub fn new_gpu(desc: TensorDesc, gpu_idx: usize, memory: GPUMemory) -> Self {
        let buf = GpuData { memory };
        Self {
            desc,
            device: DeviceId::Gpu(gpu_idx),
            buffer: Box::new(buf),
        }
    }

    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    pub fn desc_mut(&mut self) -> &mut TensorDesc {
        &mut self.desc
    }

    pub fn device(&self) -> &DeviceId {
        &self.device
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self.device, DeviceId::Cpu)
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self.device, DeviceId::Gpu(_))
    }

    /// Return length in bytes of the underlying storage.
    pub fn len_bytes(&self) -> usize {
        self.buffer.len_bytes()
    }

    pub fn read(&self) -> Box<[u8]> {
        self.buffer.read()
    }

    pub fn write(&mut self, data: &[u8]) {
        self.buffer.write(data);
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

    pub fn get_cpu_memory_slice_or_panic(&self) -> &[u8] {
        let any_mut = self.buffer.as_any();
        let cpu = any_mut
            .downcast_ref::<CpuData>()
            .expect("Tensor is not backed by CPU storage");

        &cpu.data
    }

    pub fn get_cpu_memory_mut_slice_or_panic(&mut self) -> &mut [u8] {
        let any_mut = self.buffer.as_any_mut();
        let cpu = any_mut
            .downcast_mut::<CpuData>()
            .expect("Tensor is not backed by CPU storage");

        &mut cpu.data
    }
}
