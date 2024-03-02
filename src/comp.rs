use crate::{ast::Ast, module::Module, tir::Tir, utils::slice::OwnedString, utir::Utir};
use anyhow::Result;
use codespan_reporting::files::SimpleFiles;
use std::mem::MaybeUninit;
use symbol_table::GlobalSymbol;

pub struct Compilation<'module> {
    files: SimpleFiles<GlobalSymbol, OwnedString>,
    modules: Vec<Module<'module>>,
}

impl<'module> Compilation<'module> {
    pub fn new() -> Self {
        return Self {
            files: SimpleFiles::new(),
            modules: Vec::new(),
        };
    }

    pub fn compile(&mut self) -> Result<()> {
        let options = CompilationOptions::from_args();
        let contents = std::fs::read_to_string(options.top_file.as_str())?;
        let file_id = self
            .files
            .add(options.top_file, OwnedString::from(contents));

        self.modules.push(Module::new());
        let module = &mut self.modules.last_mut().unwrap();

        let ast = Ast::parse(self.files.get(file_id)?)?;
        if options.dump_ast {
            println!("{}", ast);
            if options.exit_early {
                return Ok(());
            }
        }

        let utir = match Utir::gen(&ast) {
            Ok(utir) => utir,
            Err(fail) => return fail.report(&ast),
        };
        if options.dump_utir {
            println!("{}", utir);
            if options.exit_early {
                return Ok(());
            }
        }

        let tir = match Tir::gen(module, &utir) {
            Ok(tir) => tir,
            Err(fail) => return fail.report(&ast),
        };

        return Ok(());
    }
}

struct CompilationOptions {
    top_file: GlobalSymbol,
    exit_early: bool,
    dump_ast: bool,
    dump_utir: bool,
    dump_tir: bool,
}

impl CompilationOptions {
    pub fn from_args() -> CompilationOptions {
        let mut top_file = MaybeUninit::uninit();
        let mut found_top = false;
        let mut exit_early = false;
        let mut dump_ast = false;
        let mut dump_utir = false;
        let mut dump_tir = false;
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--exit-early" => exit_early = true,
                "--dump-ast" => dump_ast = true,
                "--dump-utir" => dump_utir = true,
                "--dump-tir" => dump_tir = true,
                _ => {
                    if found_top {
                        println!("[ERROR] Found multiple top files in arguments");
                        std::process::exit(1);
                    } else {
                        top_file.write(GlobalSymbol::from(arg));
                        found_top = true;
                    }
                }
            }
        }

        if !found_top {
            println!("[ERROR] Could not find top file in arguments");
            std::process::exit(1);
        }

        return CompilationOptions {
            top_file: unsafe { top_file.assume_init() },
            exit_early,
            dump_ast,
            dump_utir,
            dump_tir,
        };
    }
}
