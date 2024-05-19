use crate::ast::Node;
use crate::{module::decls::Decl, utils::RRC};

#[derive(Hash)]
pub struct Comb {
    owner_decl: RRC<Decl>,
    node_ptr: *const Node,
    analysis: CombAnalysis,
}

impl Comb {
    pub fn new(owner_decl: RRC<Decl>, node: &Node) -> Self {
        let node_ptr = node as *const _;
        Self {
            owner_decl,
            node_ptr,
            analysis: CombAnalysis::Queued,
        }
    }
}

impl PartialEq for Comb {
    fn eq(&self, other: &Self) -> bool {
        self.owner_decl == other.owner_decl
    }
}

impl Eq for Comb {}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum CombAnalysis {
    Queued,
    InProgress,
    FailedSema,
    Success,
}
