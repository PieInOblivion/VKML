pub type LayerId = usize;

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
