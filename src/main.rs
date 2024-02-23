mod ast;
mod auto_indenting_stream;
mod builtin;
mod comp;
mod parser;
mod sema;
mod types;
mod utils;
mod utir;
mod values;

use anyhow::Result;
use ast::Ast;
use comp::Compilation;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        println!("[ERROR] Expected ./prog <filename>");
        std::process::exit(1);
    }

    let mut compilation = Compilation::new();
    compilation.compile_all()?;

    return Ok(());
}
