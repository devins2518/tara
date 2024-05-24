use crate::{
    ast::Node, module::decls::Decl, types::Type as TaraType, utils::RRC, values::StaticMlirValue,
};
use anyhow::Result;
use std::hash::Hash;

pub struct Register {
    pub decl: RRC<Decl>,
    pub node_ptr: *const Node,
    pub ty: TaraType,
    pub def_op: StaticMlirValue,
    pub analysis: RegisterAnalysis,
}

impl Register {
    pub fn new(decl: RRC<Decl>, node: &Node, ty: TaraType, def_op: StaticMlirValue) -> Self {
        let node_ptr = node as *const _;
        Self {
            decl,
            node_ptr,
            ty,
            def_op,
            analysis: RegisterAnalysis::Unwritten,
        }
    }

    pub fn write(&mut self) -> Result<()> {
        unimplemented!()
    }
}

impl PartialEq for Register {
    fn eq(&self, other: &Self) -> bool {
        self.decl == other.decl
    }
}

impl Eq for Register {}

impl Hash for Register {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.node_ptr.hash(state);
    }
}

#[derive(PartialEq, Eq)]
pub enum RegisterAnalysis {
    Unwritten,
    Written,
}
