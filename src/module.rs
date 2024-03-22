pub mod decls;
pub mod file;
pub mod function;
pub mod namespace;
pub mod structs;
pub mod variable;

use crate::{
    ast::Ast,
    codegen::{package::Package, Codegen},
    comp::Compilation,
    module::{
        decls::{Decl, DeclStatus},
        file::File,
        namespace::Namespace,
        structs::{Struct, StructStatus},
    },
    types::Type,
    utils::{init_field, RRC},
    utir::{inst::UtirInstIdx, Utir},
    values::Value,
};
use anyhow::{bail, Result};
use core::fmt;
use kioku::Arena;
use melior::Context;
use std::{collections::HashMap, error::Error, path::PathBuf};

#[derive(Debug)]
enum FailKind {
    ParseFail,
    AstFail,
    UtirFail,
}

impl fmt::Display for FailKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for FailKind {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

pub struct Module<'comp> {
    arena: &'comp Arena,
    // Keys are fully resolved paths
    pub import_table: HashMap<PathBuf, RRC<File>>,
}

impl<'comp> Module<'comp> {
    pub fn new(comp: &'comp Compilation) -> Self {
        Self {
            arena: &comp.arena,
            import_table: HashMap::new(),
        }
    }

    pub fn analyze_main_pkg(
        &mut self,
        main_pkg_path: &str,
        exit_early: bool,
        dump_ast: bool,
        dump_utir: bool,
        dump_mlir: bool,
    ) -> Result<()> {
        let main_pkg = {
            let resolved_main_pkg_path = std::fs::canonicalize(main_pkg_path)?;
            // Get directory itself
            let src_dir = resolved_main_pkg_path.parent().unwrap().to_path_buf();
            // Get file itself
            let src_path = resolved_main_pkg_path
                .file_name()
                .unwrap()
                .to_os_string()
                .into_string()
                .unwrap();
            RRC::new(Package {
                src_dir,
                src_path,
                pkg_path: "root".to_string(),
            })
        };
        let file = self.import_pkg(main_pkg)?.file;
        self.analyze_file(file, exit_early, dump_ast, dump_utir, dump_mlir)?;
        Ok(())
    }

    fn analyze_file(
        &mut self,
        file: RRC<File>,
        exit_early: bool,
        dump_ast: bool,
        dump_utir: bool,
        dump_mlir: bool,
    ) -> Result<()> {
        {
            let f = &mut file.borrow_mut();
            self.load_file(f)?;

            self.parse(f)?;
            if dump_ast {
                println!("{}", f.ast());
                if exit_early || !(dump_utir || dump_mlir) {
                    return Ok(());
                }
            }

            match self.gen_utir(f) {
                Ok(_) => {}
                Err(_) => return Ok(()),
            }
            if dump_utir {
                println!("{}", f.utir());
                if exit_early || !dump_mlir {
                    return Ok(());
                }
            }

            let utir = f.utir();
            // self.analyze_top(utir);
            // debug_assert!(self.module.as_operation().verify());
            if dump_mlir {
                // self.module.as_operation().dump();
            }
        }

        self.sema_file(file);

        Ok(())
    }

    fn sema_file(&self, file: RRC<File>) {
        if file.borrow().root_decl.is_some() {
            return;
        }

        let (struct_obj, struct_obj_uninit) = RRC::<Struct>::new_uninit();
        let struct_ty = RRC::new(Type::Struct(struct_obj.clone()));
        let struct_val = Value::Type(struct_ty.clone());
        let ty_ty = Type::TypeType;
        let namespace = RRC::new(Namespace::new(file.clone(), struct_ty));

        let main_idx = UtirInstIdx::from(0);

        let decl_name = file.borrow().fully_qualified_path();
        let decl = RRC::new(Decl::new(
            decl_name,
            namespace.clone(),
            main_idx.into(),
            None,
        ));
        file.borrow_mut().root_decl = Some(decl.clone());

        // Initialize struct_obj
        {
            let mut uninit = struct_obj_uninit.borrow_mut();
            init_field!(uninit, owner_decl, decl.clone());
            init_field!(uninit, fields, Vec::new());
            init_field!(uninit, namespace, namespace.clone());
            init_field!(uninit, status, StructStatus::None);
            init_field!(uninit, utir_ref, UtirInstIdx::from(0).into());
            drop(uninit);
            struct_obj_uninit.init();
        }

        // Setup decl
        {
            let mut decl = decl.borrow_mut();
            decl.public = true;
            decl.ty = Some(ty_ty);
            decl.value = Some(struct_val);
            decl.status = DeclStatus::InProgress;
        }

        let file = file.borrow();
        let utir = file.utir();
        let context = Context::new();
        let mut codegen = Codegen::new(&context, utir);
        codegen.analyze_struct_decl(decl, main_idx, struct_obj);
    }

    // This is pretty dumb and expensive, rework packages to be cheaper
    fn import_pkg(&mut self, pkg: RRC<Package>) -> Result<ImportResult> {
        let full_path = pkg.borrow().full_path();
        let new = self.import_table.contains_key(&full_path);
        let file = self
            .import_table
            .entry(full_path.clone()) // TODO: Don't clone here
            .or_insert(RRC::new(File::new(
                PathBuf::from(&pkg.borrow().src_path),
                pkg.clone(),
            )))
            .clone();
        Ok(ImportResult { file, new })
    }
}

struct ImportResult {
    file: RRC<File>,
    new: bool,
}

// File related methods
impl Module<'_> {
    fn load_file(&self, file: &mut File) -> Result<()> {
        let contents = {
            use std::io::prelude::*;
            let mut fp = std::fs::File::open(file.pkg.borrow().full_path())?;
            let mut string = String::new();
            fp.read_to_string(&mut string)?;
            string
        };
        file.add_source(contents);
        Ok(())
    }

    fn parse(&self, file: &mut File) -> Result<()> {
        let source = file.source();
        let ast = match Ast::parse(source) {
            Ok(ast) => file.add_ast(ast),
            Err(_) => {
                file.fail_ast();
                return Ok(());
            }
        };
        Ok(())
    }

    fn gen_utir(&self, file: &mut File) -> Result<()> {
        let ast = file.ast();
        let utir = match Utir::gen(ast) {
            Ok(utir) => file.add_utir(utir),
            Err(fail) => {
                file.fail_utir();
                fail.report(&file)?;
                bail!(FailKind::UtirFail);
            }
        };
        Ok(())
    }
}
