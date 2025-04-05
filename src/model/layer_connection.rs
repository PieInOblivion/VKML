pub type LayerId = usize;

// Realistically, this could be a tuple (usize, usize)
// That would require slightly more knowledge to make a basic sequential model
// and having that be as easy as possible is a goal of this project.
#[derive(Clone, Debug, PartialEq)]
pub enum LayerConnection {
    DefaultOutput(LayerId),
    SpecificOutput(LayerId, usize),
}

impl LayerConnection {
    pub fn get_layerid(&self) -> LayerId {
        match self {
            LayerConnection::DefaultOutput(id) => *id,
            LayerConnection::SpecificOutput(id, _) => *id,
        }
    }

    pub fn get_outputidx(&self) -> usize {
        match self {
            LayerConnection::DefaultOutput(_) => 0,
            LayerConnection::SpecificOutput(_, idx) => *idx,
        }
    }
}
