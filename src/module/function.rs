use crate::{module::decls::Decl, utils::RRC, utir::inst::UtirInstRef, values::TypedValue};

#[derive(Hash)]
pub struct Function {
    owner_decl: RRC<Decl>,
    comptime_args: Option<Vec<TypedValue>>,
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
