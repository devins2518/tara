pub mod decls;
pub mod file;
pub mod function;
pub mod namespace;
pub mod structs;
pub mod variable;

use crate::{
    ast::Ast,
    codegen::package::Package,
    comp::Compilation,
    module::{
        decls::{CaptureScope, Decl},
        file::File,
        namespace::Namespace,
        structs::Struct,
    },
    types::Type,
    utils::RRC,
    utir::{inst::UtirInstRef, Utir},
    values::Value,
};
use anyhow::Result;
use kioku::Arena;
use std::{borrow::Borrow, collections::HashMap, mem::MaybeUninit};

pub struct Module<'comp, 'arena> {
    comp: &'comp Compilation,
    arena: &'arena Arena,
    // Keys are fully resolved paths
    pub import_table: HashMap<String, RRC<File>>,
}

impl<'comp, 'arena> Module<'comp, 'arena> {
    pub fn new(comp: &'comp Compilation, arena: &'arena Arena) -> Self {
        Self {
            comp,
            arena,
            import_table: HashMap::new(),
        }
    }

    pub fn analyze_pkg(
        &mut self,
        pkg: &'arena Package,
        exit_early: bool,
        compile_ast: bool,
        compile_utir: bool,
    ) -> Result<()> {
        let file = self.import_pkg(pkg)?.file;
        self.analyze_file(file, exit_early, compile_ast, compile_utir)?;
        // self.import_table.insert(&pkg.full_path(), file);
        Ok(())
    }

    fn analyze_file(
        &mut self,
        file: RRC<File>,
        exit_early: bool,
        compile_ast: bool,
        compile_utir: bool,
    ) -> Result<()> {
        {
            let mut f = file.borrow_mut();
            let contents = {
                use std::io::prelude::*;
                let mut fp = std::fs::File::open(&f.path)?;
                let mut string = String::new();
                fp.read_to_string(&mut string)?;
                string
            };
            f.add_source(contents);

            if exit_early && !compile_ast && !compile_utir {
                return Ok(());
            }

            let source = f.source();
            let ast = match Ast::parse(source) {
                Ok(ast) => {
                    f.add_ast(ast);
                    f.ast()
                }
                Err(_) => {
                    f.fail_ast();
                    return Ok(());
                }
            };
            if compile_ast {
                println!("{}", f.ast());
            }

            if exit_early && !compile_utir {
                return Ok(());
            }

            let utir = match Utir::gen(&ast) {
                Ok(utir) => {
                    f.add_utir(utir);
                    f.utir()
                }
                Err(fail) => {
                    f.fail_utir();
                    fail.report(&f)?;
                    return Ok(());
                }
            };
            if compile_utir {
                println!("{}", f.utir());
            }
        }

        self.sema_file(file);

        Ok(())
    }

    fn sema_file(&self, file: RRC<File>) {
        if file.borrow().root_decl.is_some() {
            return;
        }

        let struct_obj: RRC<Struct> = RRC::new_uninit();
        let struct_ty = RRC::new(Type::Struct(struct_obj));
        let struct_val = Value::Type(struct_ty.clone());
        let ty_ty = Value::TypeType;
        let namespace = Namespace::new(file, struct_ty);
    }

    fn alloc_uninit<T>(&self) -> &'arena mut T {
        self.arena
            .alloc_no_copy(unsafe { MaybeUninit::uninit().assume_init() })
    }

    // This is pretty dumb and expensive, rework packages to be cheaper
    fn import_pkg(&mut self, pkg: &'arena Package) -> Result<ImportResult> {
        let full_path = pkg.full_path();
        let new = self.import_table.contains_key(&full_path);
        let file = self
            .import_table
            .entry(full_path.clone()) // TODO: Don't clone here
            .or_insert(RRC::new(File::new(full_path)))
            .clone();
        Ok(ImportResult { file, new })
    }
}

struct ImportResult {
    file: RRC<File>,
    new: bool,
}
