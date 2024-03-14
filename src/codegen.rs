mod error;
pub mod package;
mod tld;

use crate::{
    codegen::{error::Failure, package::Package, tld::Tld},
    module::file::File,
    types::Type as TType,
    values::Value as TValue,
};
use anyhow::Result;
use kioku::Arena;
use melior::{
    ir::{BlockRef, Location, Module, Value},
    Context,
};
use std::collections::HashMap;

pub struct Codegen<'arena, 'ctx> {
    arena: &'arena Arena,
    context: Context,
    module: Module<'ctx>,
    errors: Vec<Failure>,
    resolve_queue: Vec<Tld>,
    // builder: BlockRef<'ctx, 'ctx>, Get from module.body()
    import_table: HashMap<&'arena str, TType<'arena>>,
    types: HashMap<&'arena str, TType<'arena>>,
    type_info: HashMap<TType<'arena>, TValue<'arena>>,
    main_pkg: Package<'arena>,
}

impl<'arena, 'ctx> Codegen<'arena, 'ctx> {
    pub fn new(
        arena: &'arena Arena,
        // Possibly relative path to main tara file
        main_pkg_path: &str,
        /* TODO: build_mode */
    ) -> Result<Self> {
        let context = Context::new();
        context.load_all_available_dialects();
        context.set_allow_unregistered_dialects(true);
        let loc = Location::new(&context, main_pkg_path, 0, 0);
        let module = Module::new(loc);
        let import_table = HashMap::new();
        let main_pkg = {
            let resolved_main_pkg_path = std::fs::canonicalize(main_pkg_path)?;
            let src_dir = arena.copy_str(resolved_main_pkg_path.to_str().unwrap());
            let src_path = arena.copy_str(resolved_main_pkg_path.to_str().unwrap());
            Package {
                src_dir,
                src_path,
                pkg_path: "root",
            }
        };
        Ok(Self {
            arena,
            context,
            module,
            errors: Vec::new(),
            resolve_queue: Vec::new(),
            import_table,
            types: HashMap::new(),
            type_info: HashMap::new(),
            main_pkg,
        })
    }

    pub fn analyze_root(&mut self) {
        let file = File::new(self.main_pkg.src_dir);
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
