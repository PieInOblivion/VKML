use super::r#trait::TensorStorageOps;
use std::ops::{Deref, DerefMut};

pub struct UnallocatedTensorStorage;

impl TensorStorageOps for UnallocatedTensorStorage {
    type ReadGuard<'a> = UnallocatedGuard;
    type WriteGuard<'a> = UnallocatedGuard;

    fn read_data(&self) -> Self::ReadGuard<'_> {
        panic!("Cannot read from unallocated tensor")
    }

    fn write_data(&self) -> Self::WriteGuard<'_> {
        panic!("Cannot write to unallocated tensor")
    }

    fn get_data(&self) -> Vec<u8> {
        panic!("Cannot read from unallocated tensor")
    }

    fn update_data(&self, _data: Vec<u8>) {
        panic!("Cannot update unallocated tensor")
    }

    fn size_in_bytes(&self) -> u64 {
        0
    }

    fn is_allocated(&self) -> bool {
        false
    }

    fn gpu_idx(&self) -> Option<usize> {
        None
    }

    fn location_string(&self) -> String {
        "Unallocated Tensor".to_string()
    }
}

// Dummy guard type that should never be instantiated
pub struct UnallocatedGuard;

impl Deref for UnallocatedGuard {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unreachable!("UnallocatedGuard should never be dereferenced")
    }
}

impl DerefMut for UnallocatedGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unreachable!("UnallocatedGuard should never be dereferenced")
    }
}
