use crate::{module::decls::Decl, values::Value};

#[derive(PartialEq, Eq, Hash)]
pub struct Variable<'module> {
    init: Value<'module>,
    owner_decl: &'module Decl<'module>,
    mutable: bool,
}
