mod error;
pub mod package;
mod tld;

use crate::{
    ast::Ast,
    codegen::{error::Failure, package::Package, tld::Tld},
    module::file::File,
    types::Type as TType,
    utir::{
        inst::{ContainerMember, UtirInstIdx, UtirInstRef},
        Utir,
    },
    values::Value as TValue,
};
use anyhow::Result;
use kioku::Arena;
use melior::{
    ir::{
        r#type::{IntegerType, TypeId, TypeLike},
        Block, Location, Module, Operation, Type,
    },
    Context,
};
use std::collections::HashMap;

pub struct Codegen<'arena, 'ctx> {
    arena: &'arena Arena,
    errors: Vec<Failure>,
    resolve_queue: Vec<Tld>,
    // builder: BlockRef<'ctx, 'ctx>, Get from module.body()
    module: Module<'ctx>,
    import_table: HashMap<&'arena str, TType<'arena>>,
    types: HashMap<UtirInstRef, TypeId<'ctx>>,
    type_info: HashMap<TType<'arena>, TValue<'arena>>,
    main_pkg: Package<'arena>,
}

impl<'arena, 'ctx> Codegen<'arena, 'ctx> {
    pub fn new(
        arena: &'arena Arena,
        // Possibly relative path to main tara file
        main_pkg_path: &str,
        context: Context,
        /* TODO: build_mode */
    ) -> Result<Self> {
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
            errors: Vec::new(),
            resolve_queue: Vec::new(),
            import_table,
            types: HashMap::new(),
            type_info: HashMap::new(),
            main_pkg,
        })
    }

    pub fn analyze_root(
        &mut self,
        exit_early: bool,
        dump_ast: bool,
        dump_utir: bool,
    ) -> Result<()> {
        let mut file = File::new(self.main_pkg.src_dir);

        self.load_file(&mut file)?;

        self.parse(&mut file)?;
        if dump_ast {
            println!("{}", file.ast());
            if exit_early {
                return Ok(());
            }
        }

        self.gen_utir(&mut file)?;
        if dump_utir {
            println!("{}", file.utir());
            if exit_early {
                return Ok(());
            }
        }

        let context = Context::new();
        context.load_all_available_dialects();
        context.set_allow_unregistered_dialects(true);
        let loc = Location::new(&context, &self.main_pkg.full_path(), 0, 0);
        let module = Module::new(loc);
        let utir = file.utir();
        self.analyze_top(
            utir,
            CtxBlock {
                ctx: &context,
                block: &module.body(),
            },
        );

        Ok(())
    }
}

// File related methods
impl<'arena> Codegen<'arena, '_> {
    fn load_file(&self, file: &mut File<'arena>) -> Result<()> {
        let contents = {
            use std::io::prelude::*;
            let mut fp = std::fs::File::open(file.path)?;
            let mut vec = Vec::new_in(self.arena);
            let len = fp.metadata()?.len() as usize;
            vec.reserve(len);
            unsafe {
                vec.set_len(len);
            }
            fp.read_exact(vec.as_mut_slice())?;
            // Leaking here is fine since the arena will clean it up later
            let slice = vec.leak();
            // Tara files don't necessarily need to be UTF-8 encoded
            unsafe { std::str::from_utf8_unchecked(slice) }
        };
        file.add_source(contents);
        Ok(())
    }

    fn parse(&self, file: &mut File<'arena>) -> Result<()> {
        let source = file.source();
        let ast = match Ast::parse(source) {
            Ok(ast) => self.arena.alloc_no_copy(ast),
            Err(e) => {
                file.fail_ast();
                return Err(e);
            }
        };
        file.add_ast(ast);
        Ok(())
    }

    fn gen_utir(&self, file: &mut File<'arena>) -> Result<()> {
        let ast = file.ast();
        let utir = match Utir::gen(ast) {
            Ok(utir) => self.arena.alloc_no_copy(utir),
            Err(fail) => {
                file.fail_utir();
                fail.report(&file)?;
                return Ok(());
            }
        };
        file.add_utir(utir);
        Ok(())
    }
}

#[derive(Copy, Clone)]
struct CtxBlock<'ctx, 'b> {
    ctx: &'ctx Context,
    block: &'b Block<'ctx>,
}

// Codegen related methods
impl<'ctx, 'b, 'arena> Codegen<'arena, 'ctx> {
    // Analyze the root of the tara file. This will always be Utir index 0.
    fn analyze_top(&mut self, utir: &Utir, ctx_block: CtxBlock<'ctx, 'b>) {
        let root_idx = UtirInstIdx::from(0);
        let top_level_decls = utir.get_container_decl_decls(root_idx);
        self.analyze_top_level_decls(top_level_decls, ctx_block);
    }

    fn analyze_top_level_decls(
        &mut self,
        decls: &[ContainerMember],
        ctx_block: CtxBlock<'ctx, 'b>,
    ) {
        for decl in decls {
            match decl.inst_ref {
                UtirInstRef::IntTypeU8 => {
                    let int_type = IntegerType::unsigned(ctx_block.ctx, 8);
                    self.types.insert(decl.inst_ref, int_type.id());
                }
                _ => unimplemented!(),
            }
        }
    }
}
