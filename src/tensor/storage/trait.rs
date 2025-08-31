use std::ops::{Deref, DerefMut};

/// Operations trait for individual storage implementations
pub trait TensorStorageOps: Send + Sync {
    type ReadGuard<'a>: Deref<Target = [u8]> + 'a
    where
        Self: 'a;
    type WriteGuard<'a>: DerefMut<Target = [u8]> + 'a
    where
        Self: 'a;

    fn read_data(&self) -> Self::ReadGuard<'_>;
    fn write_data(&self) -> Self::WriteGuard<'_>;
    fn get_data(&self) -> Vec<u8>;
    fn update_data(&self, data: Vec<u8>);
    fn size_in_bytes(&self) -> u64;
    fn is_allocated(&self) -> bool;
    fn gpu_idx(&self) -> Option<usize>;
    fn location_string(&self) -> String;
}
