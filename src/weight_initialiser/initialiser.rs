use onnx_extractor::Bytes;

#[derive(Default)]
pub enum Initialiser {
    #[default]
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
            Initialiser::Bytes(bytes) => bytes.to_vec().into(),
            Initialiser::BytesVec(parts) => {
                let total_len: usize = parts.iter().map(|b| b.len()).sum();
                let mut vec = Vec::with_capacity(total_len);
                for bytes in parts {
                    vec.extend_from_slice(&bytes);
                }
                vec.into()
            }
            Initialiser::OwnedBox(boxed) => boxed,
            Initialiser::OwnedVec(vec) => vec.into(),
            Initialiser::Constant(vec) => vec.into(),

            Initialiser::None => unimplemented!("None"),
            Initialiser::Xavier => unimplemented!("Xavier"),
            Initialiser::Uniform(_, _) => unimplemented!("Uniform"),
            Initialiser::He => unimplemented!("He"),
        }
    }
}
