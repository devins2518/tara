use crate::{
    module::{decls::Decl, namespace::Namespace},
    types::Type,
    utils::RRC,
    utir::inst::UtirInstRef,
};

#[derive(Hash)]
pub struct Struct {
    pub owner_decl: RRC<Decl>,
    pub fields: Vec<Field>,
    pub namespace: RRC<Namespace>,
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
#[derive(Hash)]
pub struct Field {
    ty: Type,
    offset: Option<u32>,
}
