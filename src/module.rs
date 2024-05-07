pub mod comb;
pub mod decls;
pub mod file;
pub mod function;
pub mod namespace;
pub mod structs;
pub mod tmodule;
pub mod variable;

use crate::{
    ast::Ast,
    ast_codegen::AstCodegen,
    circt,
    codegen::{package::Package, Codegen},
    comp::Compilation,
    module::{
        decls::{CaptureScope, Decl, DeclStatus},
        file::File,
        namespace::Namespace,
        structs::{Struct, StructStatus},
    },
    types::Type,
    utils::{id_arena::Id, init_field, RRC},
    utir::{
        inst::{NamedRef, UtirInstIdx},
        Utir,
    },
    values::Value,
};
use anyhow::{bail, Result};
use core::fmt;
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module as MlirModule},
    utility::register_all_dialects,
    Context,
};
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

pub struct Module {
    // Keys are fully resolved paths
    pub import_table: HashMap<PathBuf, RRC<File>>,
}

impl Module {
    pub fn new() -> Self {
        Self {
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

            /*
            self.gen_utir(f)?;
            if dump_utir {
                println!("{}", f.utir());
                if exit_early || !dump_mlir {
                    return Ok(());
                }
            }
            */
        }

        self.codegen_file(file, dump_mlir)?;
        Ok(())
    }

    fn codegen_file(&self, file: RRC<File>, dump_mlir: bool) -> Result<()> {
        if file.borrow().root_decl.is_some() {
            return Ok(());
        }

        let file = file.borrow();
        let ast = file.ast();

        let context = Context::new();
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        circt::register_all_dialects(&context);
        context.set_allow_unregistered_dialects(true);
        context.load_all_available_dialects();
        let module = MlirModule::new(Location::unknown(&context));
        let mut codegen = AstCodegen::new(&ast, &context, &module);
        codegen.gen_root()?;
        if let Err(_) = codegen.report_errors(&file) {
            return Ok(());
        }
        if dump_mlir {
            module.as_operation().dump();
        }
        Ok(())
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
impl Module {
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
        match Ast::parse(source) {
            Ok(ast) => file.add_ast(ast),
            Err(e) => {
                file.fail_ast();
                return Err(e);
            }
        };
        Ok(())
    }

    fn gen_utir(&self, file: &mut File) -> Result<()> {
        let ast = file.ast();
        match Utir::gen(ast) {
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

// Methods for codegen (i.e. namespace lookup)
impl Module {
    pub fn scan_namespace(
        &self,
        context: &Context,
        namespace: RRC<Namespace>,
        decls: &[NamedRef],
        parent_decl: RRC<Decl>,
    ) -> Result<()> {
        for decl in decls {
            self.scan_decl(context, decl, namespace.clone(), parent_decl.clone())?;
        }
        Ok(())
    }

    fn scan_decl(
        &self,
        context: &Context,
        named_ref: &NamedRef,
        namespace: RRC<Namespace>,
        parent_decl: RRC<Decl>,
    ) -> Result<()> {
        let decl_name = named_ref.name.as_str();
        // TODO: Maybe use global symbols for everything?
        let decl_name_string = decl_name.to_string();
        let decl_ref = named_ref.inst_ref;
        let top_decl_name = namespace.borrow().file.borrow().fully_qualified_path();
        if namespace.borrow().decls.get(decl_name).is_none() {
            log::debug!(
                "inserting new decl ({}) into parent_decl ({})",
                decl_name,
                parent_decl.borrow().name,
            );
            let new_decl = RRC::new(Decl::new(
                decl_name_string.clone(),
                namespace.clone(),
                decl_ref.into(),
                parent_decl.borrow().src_scope.clone(),
            ));
            // TODO: Lazy analysis here
            // TODO: export support here
            // Kinda hacky
            // let wants_analysis = parent_decl.borrow().name == top_decl_name && decl_name == "Top";
            // if wants_analysis {
            self.codegen_decl(context, new_decl.clone())?;
            // }
            namespace
                .borrow_mut()
                .decls
                .entry(decl_name_string.clone())
                .or_insert(new_decl.clone());
        }
        Ok(())
    }

    fn codegen_decl(&self, context: &Context, decl: RRC<Decl>) -> Result<()> {
        {
            decl.borrow_mut().status = DeclStatus::InProgress;
        }
        let file_scope = decl.borrow().file_scope();
        let file = file_scope.borrow();
        let utir = file.utir();

        let parent_capture_scope = decl.borrow().src_scope.clone();
        let capture_scope = RRC::new(CaptureScope::new(parent_capture_scope));
        let utir_inst = decl.borrow().utir_inst;
        let mut codegen = Codegen::new(self, &context, utir);
        let break_ref = codegen.analyze_top_level_decl(decl.clone(), utir_inst, capture_scope)?;
        let decl_ty_val = codegen.resolve_ref_value(break_ref);

        codegen.resolve_type_layout(decl_ty_val.ty.clone());

        {
            let mut decl = decl.borrow_mut();
            decl.ty = Some(decl_ty_val.ty);
            decl.value = Some(decl_ty_val.value);
            decl.status = DeclStatus::Complete;
        }

        unimplemented!()
    }
}
