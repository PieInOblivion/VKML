use super::{dataloader::SourceFormat, datasource::LabelType};

pub struct DataBatch {
    pub data: Box<[u8]>,
    pub samples_in_batch: usize,
    pub bytes_per_sample: usize,
    pub format: SourceFormat,
    pub labels: Option<Vec<LabelType>>,
    pub batch_number: usize,
}

impl DataBatch {
    pub fn to_f32(&self) -> Vec<f32> {
        let num_components = self.data.len() / self.format.bytes_per_element();

        let mut result = Vec::with_capacity(num_components);
        unsafe {
            result.set_len(num_components);
        }

        match self.format {
            SourceFormat::U8 => {
                for (i, &x) in self.data.iter().enumerate() {
                    result[i] = x as f32;
                }
            }
            SourceFormat::U16 => {
                for (i, chunk) in self.data.chunks_exact(2).enumerate() {
                    result[i] = u16::from_le_bytes([chunk[0], chunk[1]]) as f32;
                }
            }
            SourceFormat::F32 => {
                for (i, chunk) in self.data.chunks_exact(4).enumerate() {
                    result[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            _ => panic!("Unsupported colour type in to_f32 ImageBatch"),
        }

        result
    }
}
