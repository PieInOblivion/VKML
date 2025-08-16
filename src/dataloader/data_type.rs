#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DataType {
    U8,
    U16,
    F32,
}

impl DataType {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            DataType::U8 => 1,
            DataType::U16 => 2,
            DataType::F32 => 4,
        }
    }
}
