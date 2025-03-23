#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorDesc {
    dims: Vec<usize>,
}

impl TensorDesc {
    pub fn new(dims: Vec<usize>) -> Self {
        assert!(!dims.is_empty(), "Tensor dimensions cannot be empty");
        Self { dims }
    }

    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    // assuming f32 elements
    pub fn size_in_bytes(&self) -> usize {
        self.num_elements() * std::mem::size_of::<f32>()
    }

    // Get dimensions vector
    pub fn to_dims(&self) -> Vec<usize> {
        self.dims.clone()
    }

    // Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    // Reshape to new dimensions (preserving total elements)
    pub fn reshape(&mut self, new_dims: Vec<usize>) -> Result<(), String> {
        if new_dims.is_empty() {
            format!("New shape must have at least one dimension");
        }

        let new_elements: usize = new_dims.iter().product();
        if new_elements != self.num_elements() {
            format!("New shape must have the same number of elements");
        }

        self.dims = new_dims;
        Ok(())
    }

    // Check if this shape can be reshaped to another
    pub fn is_reshapable_to(&self, other: &Self) -> bool {
        self.num_elements() == other.num_elements()
    }

    // Calculate strides for row-major memory layout
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.dims.len()];
        let mut stride = 1;

        // Calculate strides from right to left (row-major)
        for i in (0..self.dims.len()).rev() {
            strides[i] = stride;
            stride *= self.dims[i];
        }

        strides
    }

    // Flatten to 1D
    pub fn flatten(&self) -> Self {
        Self {
            dims: vec![self.num_elements()],
        }
    }

    pub fn calculate_fan_in_out(&self) -> (usize, usize) {
        // For 1D tensors, assume bias vector or similar
        if self.dims.len() == 1 {
            return (1, self.dims[0]);
        }

        // First dimension is typically output features
        let out_features = self.dims[0];

        // Second dimension is typically input features
        let in_features = if self.dims.len() > 1 { self.dims[1] } else { 1 };

        // Any remaining dimensions represent the kernel/spatial dimensions
        // Calculate their product
        let kernel_size: usize = if self.dims.len() > 2 {
            self.dims[2..].iter().product()
        } else {
            1
        };

        // fan_in = input_features × kernel_size
        // fan_out = output_features × kernel_size
        (in_features * kernel_size, out_features * kernel_size)
    }
}
