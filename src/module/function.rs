use crate::{ast::Node, module::decls::Decl, utils::RRC, values::TypedValue};
use std::cmp::{Eq, PartialEq};

#[derive(Hash)]
pub struct Function {
    decl: RRC<Decl>,
    comptime_args: Option<Vec<TypedValue>>,
    node_ptr: *const Node,
    analysis: FunctionAnalysis,
}

impl Function {
    pub fn new(decl: RRC<Decl>, node: &Node) -> Self {
        let node_ptr = node as *const _;
        Self {
            decl,
            comptime_args: None,
            node_ptr,
            analysis: FunctionAnalysis::Queued,
        }
    }
}

impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.decl == other.decl
    }
}

impl Eq for Function {}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum FunctionAnalysis {
    Queued,
    InProgress,
    FailedSema,
    Success,
}
