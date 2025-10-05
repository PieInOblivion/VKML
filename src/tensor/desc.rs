use onnx_extractor::DataType;

#[derive(Clone, Debug)]
pub struct TensorDesc {
    dims: Vec<i64>,
    data_type: DataType,
}

impl TensorDesc {
    pub fn new(dims: Vec<i64>, data_type: DataType) -> Self {
        assert!(!dims.is_empty(), "Tensor dimensions cannot be empty");
        Self { dims, data_type }
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(|d| *d as usize).product()
    }

    // Size in bytes for the tensor given its DataType
    pub fn size_in_bytes(&self) -> usize {
        let elem_size = self.data_type.size_in_bytes().unwrap();
        self.num_elements() * elem_size
    }

    // Get dimensions
    pub fn dims(&self) -> &[i64] {
        &self.dims
    }

    // Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    // Reshape to new dimensions (preserving total elements)
    pub fn reshape(&mut self, new_dims: Vec<i64>) -> Result<(), String> {
        if new_dims.is_empty() {
            return Err("New shape must have at least one dimension".to_string());
        }

        let new_elements: usize = new_dims.iter().map(|d| *d as usize).product();
        if new_elements != self.num_elements() {
            return Err("New shape must have the same number of elements".to_string());
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
        Self::compute_strides(&self.dims)
    }

    // Flatten to 1D
    pub fn flatten(&self) -> Self {
        Self {
            dims: vec![self.num_elements() as i64],
            data_type: self.data_type,
        }
    }

    pub fn calculate_fan_in_out(&self) -> (usize, usize) {
        // For 1D tensors, assume bias vector or similar
        if self.dims.len() == 1 {
            return (1, self.dims[0] as usize);
        }

        // First dimension is typically output features
        let out_features = self.dims[0] as usize;

        // Second dimension is typically input features
        let in_features = if self.dims.len() > 1 {
            self.dims[1] as usize
        } else {
            1
        };

        // Any remaining dimensions represent the kernel/spatial dimensions
        // Calculate their product
        let kernel_size: usize = if self.dims.len() > 2 {
            self.dims[2..].iter().map(|d| *d as usize).product()
        } else {
            1
        };

        // fan_in = input_features × kernel_size
        // fan_out = output_features × kernel_size
        (in_features * kernel_size, out_features * kernel_size)
    }

    pub fn compute_strides(dims: &[i64]) -> Vec<usize> {
        let mut s = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * dims[i + 1] as usize;
        }
        s
    }

    pub fn broadcast_shape(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
        let ndim = a.len().max(b.len());
        let mut out = vec![1i64; ndim];
        for i in 0..ndim {
            let ai = *a.get(a.len().wrapping_sub(i + 1)).unwrap_or(&1);
            let bi = *b.get(b.len().wrapping_sub(i + 1)).unwrap_or(&1);
            if ai == bi || ai == 1 || bi == 1 {
                out[ndim - 1 - i] = ai.max(bi);
            } else {
                return None;
            }
        }
        Some(out)
    }

    pub fn broadcast_strides(src: &[i64], dst: &[i64]) -> Vec<usize> {
        let src_strides = Self::compute_strides(src);
        let mut bs = vec![0; dst.len()];
        let offset = dst.len().saturating_sub(src.len());
        for i in 0..dst.len() {
            let dim = *src.get(i.wrapping_sub(offset)).unwrap_or(&1) as usize;
            let stride = *src_strides.get(i.wrapping_sub(offset)).unwrap_or(&0);
            bs[i] = if dim == 1 { 0 } else { stride };
        }
        bs
    }

    pub fn unravel(idx: usize, dims: &[i64]) -> Vec<usize> {
        let mut rem = idx;
        let strides = Self::compute_strides(dims);
        dims.iter()
            .enumerate()
            .map(|(i, _)| {
                let c = rem / strides[i];
                rem %= strides[i];
                c
            })
            .collect()
    }

    pub fn offset(idxs: &[usize], strides: &[usize]) -> usize {
        idxs.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
    }
}
