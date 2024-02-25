use crate::{
    ast::Ast,
    types::Type,
    utils::slice::{OwnedSlice, OwnedString},
    utir::Utir,
};
use anyhow::Result;
use codespan_reporting::files::SimpleFiles;
use internment::Arena;
use std::mem::MaybeUninit;
use symbol_table::GlobalSymbol;

pub struct Compilation<'a> {
    files: SimpleFiles<GlobalSymbol, OwnedString>,
    types: Arena<Type<'a>>,
}

impl<'a> Compilation<'a> {
    pub fn new() -> Self {
        return Self {
            files: SimpleFiles::new(),
            types: Arena::new(),
        };
    }

    pub fn compile(&mut self) -> Result<()> {
        let options = CompilationOptions::from_args();
        let contents = std::fs::read_to_string(options.top_file.as_str())?;
        let file_id = self
            .files
            .add(options.top_file, OwnedString::from(contents));

        let ast = Ast::parse(self.files.get(file_id)?)?;
        if options.dump_ast {
            println!("{}", ast);
        }

        let utir = match Utir::gen(&ast) {
            Ok(utir) => utir,
            Err(fail) => return fail.report(&ast),
        };
        if options.dump_utir {
            println!("{}", utir);
        }
        return Ok(());
    }
}

struct CompilationOptions {
    top_file: GlobalSymbol,
    dump_ast: bool,
    dump_utir: bool,
}

impl CompilationOptions {
    pub fn from_args() -> CompilationOptions {
        let mut top_file = MaybeUninit::uninit();
        let mut found_top = false;
        let mut dump_ast = false;
        let mut dump_utir = false;
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--dump-ast" => dump_ast = true,
                "--dump-utir" => dump_utir = true,
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
            dump_ast,
            dump_utir,
        };
    }
}
