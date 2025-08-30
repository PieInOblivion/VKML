use onnx_extractor::DataType;

pub struct DataBatch {
    data: Box<[u8]>,
    data_type: DataType,
}

impl DataBatch {
    pub fn new(size: usize, data_type: DataType) -> Self {
        Self {
            data: vec![0u8; size].into_boxed_slice(),
            data_type,
        }
    }

    pub fn from_bytes(bytes: Vec<u8>, data_type: DataType) -> Self {
        Self {
            data: bytes.into_boxed_slice(),
            data_type,
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn num_elements(&self) -> usize {
        self.data.len() / self.data_type.size_in_bytes().unwrap_or(4)
    }

    pub fn to_f32(&self) -> Vec<f32> {
        match self.data_type {
            DataType::Uint8 => self.data.iter().map(|&x| x as f32).collect(),
            DataType::Uint16 => self
                .data
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as f32)
                .collect(),
            DataType::Float => self
                .data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            other => panic!(
                "Conversion from data type {:?} to f32 not implemented",
                other
            ),
        }
    }

    pub fn to_data_type(&mut self, target_type: DataType) {
        if self.data_type == target_type {
            return;
        }

        let converted_bytes = match (self.data_type, target_type) {
            // To Float conversions
            (DataType::Uint8, DataType::Float) => {
                let mut result = Vec::with_capacity(self.data.len() * 4);
                for &byte in self.data.iter() {
                    let f = byte as f32;
                    result.extend_from_slice(&f.to_le_bytes());
                }
                result
            }
            (DataType::Uint16, DataType::Float) => {
                let mut result = Vec::with_capacity((self.data.len() / 2) * 4);
                for chunk in self.data.chunks_exact(2) {
                    let u = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let f = u as f32;
                    result.extend_from_slice(&f.to_le_bytes());
                }
                result
            }
            _ => panic!(
                "Conversion from {:?} to {:?} not implemented",
                self.data_type, target_type
            ),
        };

        self.data = converted_bytes.into_boxed_slice();
        self.data_type = target_type;
    }

    /*
    pub fn write_at(&mut self, offset: usize, src: &[u8]) {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.0.as_mut_ptr().add(offset), src.len());
        }
    }
    */
}
