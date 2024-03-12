use crate::{module::decls::Decl, utir::inst::UtirInstRef, values::TypedValue};

#[derive(PartialEq, Eq, Hash)]
pub struct Function<'module> {
    owner_decl: &'module Decl<'module>,
    comptime_args: Option<Vec<TypedValue<'module>>>,
    utir_inst: UtirInstRef,
    analysis: FunctionAnalysis,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum FunctionAnalysis {
    Queued,
    InProgress,
    FailedSema,
    Success,
}
