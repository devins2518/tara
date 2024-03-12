mod error;

use crate::{
    codegen::error::Failure,
    module::Module,
    tables::Tables,
    utir::{
        inst::{UtirInst, UtirInstIdx, UtirInstRef},
        Utir,
    },
};
use melior::Context;

pub struct Codegen<'comp> {
    module: Module<'comp>,
    tables: Tables,
    context: Context,
}

impl<'comp> Codegen<'comp> {
    pub fn new(module: Module<'comp>) -> Self {
        let context = Context::new();
        context.load_all_available_dialects();
        context.set_allow_unregistered_dialects(true);
        return Self {
            module,
            tables: Tables::new(),
            context,
        };
    }

    /*
    pub fn gen(&mut self) -> Result<(), Failure> {
        let top_ref = self
            .module
            .get_decl(UtirInstIdx::from(0), "Top")
            .ok_or(Failure::TopNotFound)?;
        // TODO:
        // self.symbol_table.insert("root.Top", top_ref);
        let top = top_ref.to_inst().ok_or(Failure::TopNotModule)?;
        self.gen_module(top);
        Ok(())
    }

    // Walks through a single layer of a struct generating any hardware constructs it passes.
    fn walk_struct_hw(&mut self, decl: UtirInstIdx<'utir>) -> Result<(), Failure> {
        let decls = self.utir.get_container_decl_decls(decl);
        // for decl in decls {
        //     let idx = decl.inst_ref;
        //     self.try_gen_module(idx);
        // }
        unimplemented!();
    }

    fn try_gen_module(&mut self, maybe_module: UtirInstRef) -> Result<(), Failure> {
        if let Some(inst) = maybe_module.to_inst() {
            self.gen_module(inst);
        }
        return Ok(());
    }

    fn gen_module(&mut self, module: UtirInstIdx<'utir>) -> Option<()> {
        let module_decl = match self.utir.get_inst(module) {
            UtirInst::ModuleDecl(inner) => inner,
            _ => return None,
        };
        let node = self.utir.get_node(module_decl.node_idx);
        // let entity = UnitData::new(UnitKind::Entity, unimplemented!(), Signature::new());

        // for field_idx in 0..module_idx {}
        Some(())
    }

    pub fn dump_module(&self) {
        // println!("{}", self.module.dump());
    }
    */
}
