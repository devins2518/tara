use crate::module::Module;
use anyhow::Result;
use kioku::Arena;
use std::mem::MaybeUninit;
use symbol_table::GlobalSymbol;

pub struct Compilation {
    pub arena: Arena,
}

impl Compilation {
    pub fn new() -> Self {
        Self {
            arena: Arena::new(),
        }
    }

    pub fn compile(&mut self) -> Result<()> {
        let options = CompilationOptions::from_args();

        let mut module = Module::new();
        module.analyze_main_pkg(
            options.top_file.as_str(),
            options.exit_early,
            options.dump_ast,
            options.dump_utir,
            options.dump_mlir,
            options.dump_verilog,
        )?;

        Ok(())
    }

    pub fn alloc<'comp, T>(&'comp self, val: T) -> &'comp mut T {
        self.arena.alloc_no_copy(val)
    }
}

struct CompilationOptions {
    top_file: GlobalSymbol,
    exit_early: bool,
    dump_ast: bool,
    dump_utir: bool,
    dump_mlir: bool,
    dump_verilog: bool,
}

impl CompilationOptions {
    pub fn from_args() -> CompilationOptions {
        let mut top_file = MaybeUninit::uninit();
        let mut found_top = false;
        let mut exit_early = false;
        let mut dump_ast = false;
        let mut dump_utir = false;
        let mut dump_mlir = false;
        let mut dump_verilog = false;
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--exit-early" => exit_early = true,
                "--dump-ast" => dump_ast = true,
                "--dump-utir" => dump_utir = true,
                "--dump-mlir" => dump_mlir = true,
                "--dump-verilog" => dump_verilog = true,
                _ => {
                    if arg.starts_with("--") {
                        println!("[ERROR] Unknown argument {}", arg);
                        std::process::exit(1);
                    }
                    if found_top {
                        println!("[ERROR] Found multiple top files in arguments");
                        std::process::exit(1);
                    }
                    top_file.write(GlobalSymbol::from(arg));
                    found_top = true;
                }
            }
        }

        if !found_top {
            println!("[ERROR] Could not find top file in arguments");
            std::process::exit(1);
        }

        CompilationOptions {
            top_file: unsafe { top_file.assume_init() },
            exit_early,
            dump_ast,
            dump_utir,
            dump_mlir,
            dump_verilog,
        }
    }
}
