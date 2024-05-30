pub mod comb;
pub mod decls;
pub mod file;
pub mod function;
pub mod namespace;
pub mod package;
pub mod register;
pub mod structs;
pub mod tmodule;
pub mod variable;

use crate::{
    ast::Ast,
    ast_codegen::AstCodegen,
    circt,
    module::{file::File, package::Package},
    utils::RRC,
    utir::Utir,
};
use anyhow::{bail, Result};
use core::fmt;
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module as MlirModule},
    pass::{transform, PassManager},
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
        dump_verilog: bool,
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
        self.analyze_file(
            file,
            exit_early,
            dump_ast,
            dump_utir,
            dump_mlir,
            dump_verilog,
        )?;
        Ok(())
    }

    fn analyze_file(
        &mut self,
        file: RRC<File>,
        exit_early: bool,
        dump_ast: bool,
        dump_utir: bool,
        dump_mlir: bool,
        dump_verilog: bool,
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
        }

        self.codegen_file(file, dump_mlir, dump_verilog)?;
        Ok(())
    }

    fn codegen_file(&self, file: RRC<File>, dump_mlir: bool, dump_verilog: bool) -> Result<()> {
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
        circt::register_all_passes();
        context.set_allow_unregistered_dialects(true);
        context.load_all_available_dialects();
        let mut module = MlirModule::new(Location::unknown(&context));
        let mut codegen = AstCodegen::new(&ast, &context, &module);
        codegen.gen_root()?;
        if let Err(_) = codegen.report_errors(&file) {
            return Ok(());
        }
        if dump_mlir {
            if dump_verilog {
                // TODO: Need to add seq_to_sv pass
                let pass_manager = PassManager::new(&context);
                pass_manager.add_pass(transform::create_canonicalizer());
                pass_manager.run(&mut module)?;

                circt::export_verilog(&module);
            }
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
