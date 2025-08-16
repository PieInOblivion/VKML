use std::sync::atomic::{AtomicU64, Ordering};

pub struct MemoryTracker {
    maximum: u64,
    current: AtomicU64,
}

// This implementation doesn't require a mutable reference to update
// The trade off of checking after the change is that there's only one operation, so no race conditions

impl MemoryTracker {
    pub fn new(maximum: u64) -> Self {
        Self {
            maximum,
            current: AtomicU64::new(0),
        }
    }

    pub fn allocate(&self, size: u64) {
        let prev = self.current.fetch_add(size, Ordering::SeqCst);
        let new = match prev.checked_add(size) {
            Some(v) => v,
            None => {
                self.current.fetch_sub(size, Ordering::SeqCst);
                panic!(
                    "Memory allocation would overflow: current {} + size {}",
                    prev, size
                );
            }
        };

        if new > self.maximum {
            self.current.fetch_sub(size, Ordering::SeqCst);
            panic!(
                "Memory limit exceeded: tried to allocate {} bytes when {} of {} bytes are used",
                size, prev, self.maximum
            );
        }
    }

    pub fn deallocate(&self, size: u64) {
        self.current.fetch_sub(size, Ordering::SeqCst);
    }

    pub fn get_current(&self) -> u64 {
        self.current.load(Ordering::SeqCst)
    }

    pub fn get_maximum(&self) -> u64 {
        self.maximum
    }

    pub fn get_available(&self) -> u64 {
        self.maximum - self.get_current()
    }
}
