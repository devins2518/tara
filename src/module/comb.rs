use crate::{module::decls::Decl, utils::RRC, utir::inst::UtirInstRef};

#[derive(Hash)]
pub struct Comb {
    owner_decl: RRC<Decl>,
    utir_inst: UtirInstRef,
    analysis: CombAnalysis,
}

impl Comb {
    pub fn new(owner_decl: RRC<Decl>, utir_inst: UtirInstRef) -> Self {
        Self {
            owner_decl,
            utir_inst,
            analysis: CombAnalysis::Queued,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum CombAnalysis {
    Queued,
    InProgress,
    FailedSema,
    Success,
}
