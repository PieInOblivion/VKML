use std::any::Any;

/// Minimal tensor storage trait for unified tensor data (no locks, no Option)
pub trait TensorData: Send + Sync {
    /// Return length in bytes of the underlying storage.
    /// For unallocated buffers this will panic; callers should consult the Tensor's desc for size.
    fn len_bytes(&self) -> usize;

    /// Read entire buffer into a host Vec<u8>. CPU implementation may avoid allocation in future.
    fn read(&self) -> Vec<u8>;

    /// Write entire buffer from host data. Length must match.
    fn write(&mut self, data: &[u8]);

    /// Allow runtime downcast from trait object to concrete type
    fn as_any(&self) -> &dyn Any;
}
