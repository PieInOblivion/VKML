use onnx_extractor::Bytes;

pub enum Initialiser {
    None,
    Bytes(Bytes),
    BytesVec(Vec<Bytes>),
    OwnedBox(Box<[u8]>),
    OwnedVec(Vec<u8>),
    Constant(Vec<u8>),
    Xavier,
    Uniform(f32, f32),
    He,
}

impl Initialiser {
    pub fn as_slice(&self) -> Option<&[u8]> {
        match self {
            Initialiser::Bytes(bytes) => Some(bytes.as_ref()),
            Initialiser::OwnedBox(boxed) => Some(boxed.as_ref()),
            Initialiser::OwnedVec(vec) => Some(vec.as_ref()),
            _ => None,
        }
    }

    // consumes self
    pub fn into_cpu_buffer(self) -> Box<[u8]> {
        match self {
            Initialiser::Bytes(bytes) => bytes.to_vec().into_boxed_slice(),
            Initialiser::OwnedBox(boxed) => boxed,
            Initialiser::OwnedVec(vec) => vec.into_boxed_slice(),
            Initialiser::BytesVec(parts) => {
                let total_len: usize = parts.iter().map(|b| b.len()).sum();
                let mut vec = Vec::with_capacity(total_len);
                for bytes in parts {
                    vec.extend_from_slice(&bytes);
                }
                vec.into_boxed_slice()
            }
            Initialiser::None => unreachable!("None case should mean caller makes vec set to 0"),
            _ => panic!("into_cpu_buffer called on non-data initialiser variant"),
        }
    }
}
