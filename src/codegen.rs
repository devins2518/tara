mod error;
pub mod package;
mod tld;

use crate::{
    circt::hw::HWModuleOperationBuilder,
    codegen::error::Failure,
    module::{decls::Decl, function::Function, structs::Struct},
    types::Type as TaraType,
    utils::RRC,
    utir::{
        inst::{ContainerMember, UtirInst, UtirInstIdx},
        Utir,
    },
    values::Value as TaraValue,
};
use anyhow::Result;
use melior::{
    ir::{attribute::StringAttribute, Location, Module as MlirModule, Region},
    Context,
};
use std::collections::HashMap;

pub struct Codegen<'cg> {
    pub ctx: &'cg Context,
    pub errors: Vec<Failure>,
    pub module: MlirModule<'cg>,
    pub type_info: HashMap<TaraType, TaraValue>,
    pub utir: &'cg Utir,
    pub func: Option<RRC<Function>>,
}

impl<'ctx> Codegen<'ctx> {
    pub fn new(
        ctx: &'ctx Context,
        utir: &'ctx Utir,
        /* TODO: build_mode */
    ) -> Self {
        let loc = Location::unknown(ctx);
        let module = MlirModule::new(loc);
        Self {
            ctx,
            module,
            errors: Vec::new(),
            type_info: HashMap::new(),
            utir,
            func: None,
        }
    }

    pub fn analyze_struct_decl(
        &mut self,
        decl: RRC<Decl>,
        utir_idx: UtirInstIdx,
        struct_obj: RRC<Struct>,
    ) -> Result<()> {
        unimplemented!()
    }
}

// Codegen related methods
impl<'cg> Codegen<'cg> {
    // Analyze the root of the tara file. This will always be Utir index 0.
    fn analyze_top(&mut self) {
        let root_idx = UtirInstIdx::from(0);
        let top_level_decls = self.utir.get_container_decl_decls(root_idx);
        self.analyze_top_level_decls(top_level_decls);
    }

    fn analyze_top_level_decls(&mut self, decls: &[ContainerMember]) {
        for decl in decls {
            match decl.inst_ref {
                _ => {
                    let idx = decl.inst_ref.to_inst().unwrap();
                    match self.utir.get_inst(idx) {
                        UtirInst::ModuleDecl(_) => {
                            self.generate_module(idx, decl.name.as_str());
                        }
                        UtirInst::CombDecl(_) => {
                            self.generate_comb(idx, decl.name.as_str());
                        }
                        _ => {
                            println!("unhandled {}: {}", decl.name, decl.inst_ref);
                            unimplemented!()
                        }
                    }
                }
            }
        }
    }

    fn generate_module(&mut self, idx: UtirInstIdx, name: &str) {
        let fields = self.utir.get_container_decl_fields(idx);
        let decls = self.utir.get_container_decl_decls(idx);
        println!(
            "generting %{} module {} fields {} decls",
            u32::from(idx),
            fields.len(),
            decls.len()
        );
        self.analyze_top_level_decls(decls);
        let region = Region::new();
        let module_builder = {
            let loc = Location::unknown(self.ctx);
            HWModuleOperationBuilder::new(self.ctx, loc)
                .sym_name(StringAttribute::new(self.ctx, name))
        };
        // TODO: generate fields
    }

    fn generate_comb(&mut self, idx: UtirInstIdx, name: &str) {
        unimplemented!()
    }
}
