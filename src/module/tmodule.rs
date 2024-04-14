use crate::{
    module::{decls::Decl, namespace::Namespace},
    utils::RRC,
};

#[derive(Hash)]
pub struct TModule {
    pub owner_decl: RRC<Decl>,
    pub combs: Vec<RRC<Decl>>,
    pub namespace: RRC<Namespace>,
}

impl TModule {
    pub fn new(owner_decl: RRC<Decl>, namespace: RRC<Namespace>) -> Self {
        TModule {
            owner_decl,
            combs: Vec::new(),
            namespace,
        }
    }
}
