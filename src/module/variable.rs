use crate::{module::decls::Decl, utils::RRC, values::Value};

#[derive(Hash)]
pub struct Variable {
    init: Value,
    owner_decl: RRC<Decl>,
    mutable: bool,
}
