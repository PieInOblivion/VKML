use super::{storage::TensorStorage, tensor_desc::TensorDesc};
use crate::gpu::gpu_memory::GPUMemory;

pub struct ComputeTensor {
    pub desc: TensorDesc,
    pub data: TensorStorage,
}

impl ComputeTensor {
    pub fn new_cpu(desc: TensorDesc, data: Vec<u8>) -> Self {
        Self {
            desc,
            data: TensorStorage::new_cpu(data),
        }
    }

    pub fn new_cpu_zeros(desc: TensorDesc) -> Self {
        let num_elements = desc.num_elements();
        Self {
            desc,
            data: TensorStorage::new_cpu(vec![0; num_elements * std::mem::size_of::<f32>()]),
        }
    }

    pub fn new_gpu(desc: TensorDesc, gpu_idx: usize, memory: GPUMemory) -> Self {
        Self {
            desc,
            data: TensorStorage::new_gpu(gpu_idx, memory),
        }
    }

    pub fn new_unallocated(desc: TensorDesc) -> Self {
        Self {
            desc,
            data: TensorStorage::new_unallocated(),
        }
    }
}
