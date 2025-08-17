/// Simplified base trait for all tensor storage implementations
pub trait TensorStorage: Send + Sync {
    /// Read all data from storage as f32 vector
    fn get_data(&self) -> Vec<f32>;
    
    /// Update storage with new data - panics on size mismatch or failure
    fn update_data(&self, data: Vec<f32>);
    
    /// Get size in bytes
    fn size_in_bytes(&self) -> u64;
    
    /// Check if storage is allocated (not unallocated placeholder)
    fn is_allocated(&self) -> bool;
    
    /// Get GPU index if this is GPU storage, None for CPU/other storage
    fn gpu_idx(&self) -> Option<usize>;
    
    /// Get human-readable location description
    fn location_string(&self) -> String;
}