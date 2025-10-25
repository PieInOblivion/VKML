use onnx_extractor::Bytes;

pub enum Initialiser<'a> {
    None,
    Ref(&'a [u8]),
    Bytes(Bytes),
    BytesVec(Vec<Bytes>),
    OwnedBox(Box<[u8]>),
    OwnedVec(Vec<u8>),
    Constant(f32),
    Xavier,
    Uniform(f32, f32),
    He,
}

impl<'a> Initialiser<'a> {
    /// Get data as slice if available
    pub fn as_slice(&self) -> Option<&[u8]> {
        match self {
            Initialiser::Bytes(bytes) => Some(bytes.as_ref()),
            Initialiser::OwnedBox(boxed) => Some(boxed.as_ref()),
            Initialiser::OwnedVec(vec) => Some(vec.as_ref()),
            _ => None,
        }
    }
}
