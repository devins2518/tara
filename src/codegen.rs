mod error;
pub mod package;
mod tld;

use crate::{
    circt::hw::HWModuleOperationBuilder,
    codegen::{error::Failure, package::Package, tld::Tld},
    types::Type as TaraType,
    utir::{
        inst::{ContainerMember, UtirInst, UtirInstIdx, UtirInstRef},
        Utir,
    },
    values::Value as TaraValue,
};
use anyhow::Result;
use kioku::Arena;
use melior::{
    ir::{
        attribute::StringAttribute,
        r#type::{IntegerType, TypeId, TypeLike},
        Location, Module, Region,
    },
    Context,
};
use std::collections::HashMap;

pub struct Codegen<'arena, 'ctx> {
    arena: &'arena Arena,
    ctx: &'ctx Context,
    errors: Vec<Failure>,
    resolve_queue: Vec<Tld>,
    // builder: BlockRef<'ctx, 'ctx>, Get from module.body()
    module: Module<'ctx>,
    import_table: HashMap<&'arena str, TaraType>,
    types: HashMap<UtirInstRef, TypeId<'ctx>>,
    type_info: HashMap<TaraType, TaraValue>,
    main_pkg: Package,
    // curr_module: Option<>
}

impl<'arena, 'ctx> Codegen<'arena, 'ctx> {
    pub fn new(
        arena: &'arena Arena,
        // Possibly relative path to main tara file
        main_pkg_path: &str,
        ctx: &'ctx Context,
        /* TODO: build_mode */
    ) -> Result<Self> {
        let loc = Location::unknown(ctx);
        let module = Module::new(loc);
        let import_table = HashMap::new();
        let main_pkg = {
            let resolved_main_pkg_path = std::fs::canonicalize(main_pkg_path)?;
            // Get directory itself
            let src_dir = resolved_main_pkg_path
                .parent()
                .unwrap()
                .to_path_buf()
                .into_os_string()
                .into_string()
                .unwrap();
            // Get file itself
            let src_path = resolved_main_pkg_path
                .file_name()
                .unwrap()
                .to_os_string()
                .into_string()
                .unwrap();
            Package {
                src_dir,
                src_path,
                pkg_path: "root".to_string(),
            }
        };
        Ok(Self {
            arena,
            module,
            errors: Vec::new(),
            resolve_queue: Vec::new(),
            import_table,
            types: HashMap::new(),
            type_info: HashMap::new(),
            main_pkg,
            ctx,
        })
    }

    /*
    pub fn analyze_root(
        &mut self,
        exit_early: bool,
        dump_ast: bool,
        dump_utir: bool,
        dump_mlir: bool,
    ) -> Result<()> {
        let mut file = File::new(self.main_pkg.full_path());

        self.load_file(&mut file)?;

        self.parse(&mut file)?;
        if dump_ast {
            println!("{}", file.ast());
            if exit_early || !(dump_utir || dump_mlir) {
                return Ok(());
            }
        }

        match self.gen_utir(&mut file) {
            Ok(_) => {}
            Err(_) => return Ok(()),
        }
        if dump_utir {
            println!("{}", file.utir());
            if exit_early || !dump_mlir {
                return Ok(());
            }
        }

        let utir = file.utir();
        self.analyze_top(utir);
        debug_assert!(self.module.as_operation().verify());
        if dump_mlir {
            self.module.as_operation().dump();
        }

        Ok(())
    }
    */
}

// Codegen related methods
impl<'ctx, 'arena> Codegen<'arena, 'ctx> {
    // Analyze the root of the tara file. This will always be Utir index 0.
    fn analyze_top(&mut self, utir: &Utir) {
        let root_idx = UtirInstIdx::from(0);
        let top_level_decls = utir.get_container_decl_decls(root_idx);
        self.analyze_top_level_decls(utir, top_level_decls);
    }

    fn analyze_top_level_decls(&mut self, utir: &Utir, decls: &[ContainerMember]) {
        for decl in decls {
            match decl.inst_ref {
                UtirInstRef::IntTypeU8 => {
                    let int_type = IntegerType::unsigned(&self.ctx, 8);
                    self.types.insert(decl.inst_ref, int_type.id());
                }
                UtirInstRef::IntTypeU16 => {
                    let int_type = IntegerType::unsigned(&self.ctx, 16);
                    self.types.insert(decl.inst_ref, int_type.id());
                }
                _ => {
                    let idx = decl.inst_ref.to_inst().unwrap();
                    match utir.get_inst(idx) {
                        UtirInst::ModuleDecl(_) => {
                            self.generate_module(utir, idx, decl.name.as_str());
                        }
                        UtirInst::CombDecl(_) => {
                            self.generate_comb(utir, idx, decl.name.as_str());
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

    fn generate_module(&mut self, utir: &Utir, idx: UtirInstIdx, name: &str) {
        let fields = utir.get_container_decl_fields(idx);
        let decls = utir.get_container_decl_decls(idx);
        println!(
            "generting %{} module {} fields {} decls",
            u32::from(idx),
            fields.len(),
            decls.len()
        );
        self.analyze_top_level_decls(utir, decls);
        let region = Region::new();
        let module_builder = {
            let loc = Location::unknown(self.ctx);
            HWModuleOperationBuilder::new(self.ctx, loc)
                .sym_name(StringAttribute::new(self.ctx, name))
        };
        // TODO: generate fields
    }

    fn generate_comb(&mut self, utir: &Utir, idx: UtirInstIdx, name: &str) {
        unimplemented!()
    }
}
