use crate::{
    module::{decls::Decl, namespace::Namespace},
    types::Type,
    utir::inst::UtirInstRef,
};

#[derive(PartialEq, Eq, Hash)]
pub struct Struct<'module> {
    owner_decl: &'module Decl<'module>,
    fields: Vec<Field<'module>>,
    namespace: Namespace<'module>,
    utir_ref: UtirInstRef,
    status: StructStatus,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum StructStatus {
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
