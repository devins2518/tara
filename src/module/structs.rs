use crate::{
    module::{decls::Decl, namespace::Namespace},
    types::Type,
    utir::inst::UtirInstRef,
};

#[derive(PartialEq, Eq, Hash)]
pub struct Struct<'module> {
    pub owner_decl: &'module Decl<'module>,
    pub fields: Vec<Field<'module>>,
    pub namespace: Namespace<'module>,
    pub utir_ref: UtirInstRef,
    pub status: StructStatus,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum StructStatus {
    None,
    FieldTypeWip,
    HaveFieldTypes,
    LayoutWip,
    HaveLayout,
    FullyResolvedWip,
    FullyResolved,
}

// TODO: default values
#[derive(PartialEq, Eq, Hash)]
pub struct Field<'module> {
    ty: Type<'module>,
    offset: Option<u32>,
}
