use crate::codegen::Codegen;
use anyhow::Result;
use kioku::Arena;
use std::mem::MaybeUninit;
use symbol_table::GlobalSymbol;

pub struct Compilation {
    arena: Arena,
}

impl Compilation {
    pub fn new() -> Self {
        return Self {
            arena: Arena::new(),
        };
    }

    pub fn compile(&mut self) -> Result<()> {
        let options = CompilationOptions::from_args();

        let codegen_arena = Arena::new();
        let mut codegen = Codegen::new(&codegen_arena, options.top_file.as_str())?;
        codegen.analyze_root(options.exit_early, options.dump_ast, options.dump_utir)?;

        return Ok(());
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
}

impl CompilationOptions {
    pub fn from_args() -> CompilationOptions {
        let mut top_file = MaybeUninit::uninit();
        let mut found_top = false;
        let mut exit_early = false;
        let mut dump_ast = false;
        let mut dump_utir = false;
        let mut dump_mlir = false;
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--exit-early" => exit_early = true,
                "--dump-ast" => dump_ast = true,
                "--dump-utir" => dump_utir = true,
                "--dump-mlir" => dump_mlir = true,
                _ => {
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
        }
    }
}
