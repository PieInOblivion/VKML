use std::cell::UnsafeCell;

use crate::tensor::tensor::Tensor;

pub struct TensorCell {
    tensor: UnsafeCell<Tensor>,
}

unsafe impl Sync for TensorCell {}

impl TensorCell {
    pub fn new(t: Tensor) -> Self {
        Self {
            tensor: UnsafeCell::new(t),
        }
    }

    pub unsafe fn as_ref(&self) -> &Tensor {
        unsafe { &*self.tensor.get() }
    }

    // safety: returns mutable ref via UnsafeCell; caller must ensure exclusive access
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_mut(&self) -> &mut Tensor {
        unsafe { &mut *self.tensor.get() }
    }
}
