use std::path::PathBuf;

pub enum DataSource {
    ImageDirectory {
        path: PathBuf,
        recursive: bool,
    },
    ImageCSV {
        path: PathBuf,
        image_column: String,
        label_columns: Option<Vec<String>>,
        has_header: bool,
    },
}

pub enum LabelType {
    SingleClass(String),
    MultiClass(Vec<String>),
    Continuous(Vec<f32>),
    Raw,
    None,
}

// TRAIT BASED

// Label support, always optional
// Labels all need to end up as f32
// label type for directory needs to be regex and then have a type as well
// eg, struct on how to format label, raw, f32 normalised (with optional range, otherwise need to read whole dataset), one hot
// So allow normalising as option. Requires knowing the maximum, so reading all the data, or allow user to tell us, have both options
// one hot encoding support etc
// Maintain current dataloader for images multi threaded abililty. generating deterministic batches, as iterator etc
// dataloder confil will need changing
// Threadpool always requirement
