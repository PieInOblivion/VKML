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
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Initialiser::Bytes(bytes) => bytes.as_ref(),
            Initialiser::OwnedBox(boxed) => boxed.as_ref(),
            Initialiser::OwnedVec(vec) => vec.as_ref(),
            Initialiser::Constant(vec) => vec.as_ref(),

            Initialiser::None => unimplemented!("None"),
            Initialiser::BytesVec(_) => unimplemented!("BytesVec"),
            Initialiser::Xavier => unimplemented!("Xavier"),
            Initialiser::Uniform(_, _) => unimplemented!("Uniform"),
            Initialiser::He => unimplemented!("He"),
        }
    }

    // consumes self
    pub fn into_cpu_buffer(self) -> Box<[u8]> {
        match self {
            Initialiser::Bytes(bytes) => bytes.to_vec().into_boxed_slice(),
            Initialiser::BytesVec(parts) => {
                let total_len: usize = parts.iter().map(|b| b.len()).sum();
                let mut vec = Vec::with_capacity(total_len);
                for bytes in parts {
                    vec.extend_from_slice(&bytes);
                }
                vec.into_boxed_slice()
            }
            Initialiser::OwnedBox(boxed) => boxed,
            Initialiser::OwnedVec(vec) => vec.into_boxed_slice(),
            Initialiser::Constant(vec) => vec.into_boxed_slice(),

            Initialiser::None => unimplemented!("None"),
            Initialiser::Xavier => unimplemented!("Xavier"),
            Initialiser::Uniform(_, _) => unimplemented!("Uniform"),
            Initialiser::He => unimplemented!("He"),
        }
    }
}
