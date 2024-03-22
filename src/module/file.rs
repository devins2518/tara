use crate::codegen::package::Package;
use crate::utils::RRC;
use crate::{ast::Ast, module::decls::Decl, utir::Utir};
use codespan_reporting::files::Files;
use std::path::PathBuf;

pub struct File {
    // Relative to main package
    pub path: PathBuf,
    status: FileStatus,
    source: Option<String>,
    ast: Option<Ast>,
    utir: Option<Utir>,
    pub root_decl: Option<RRC<Decl>>,
    pub pkg: RRC<Package>,
}

impl File {
    pub fn new(path: PathBuf, pkg: RRC<Package>) -> Self {
        Self {
            path,
            status: FileStatus::Unloaded,
            source: None,
            ast: None,
            utir: None,
            root_decl: None,
            pkg,
        }
    }

    // Asserts that source is loaded
    pub fn source(&self) -> &str {
        self.source.as_ref().unwrap()
    }
    pub fn add_source(&mut self, source: String) {
        self.source = Some(source);
        self.status = FileStatus::Loaded;
    }

    pub fn ast(&self) -> &Ast {
        self.ast.as_ref().unwrap()
    }
    pub fn add_ast(&mut self, ast: Ast) {
        self.ast = Some(ast);
        self.status = FileStatus::ParseSucceed;
    }

    pub fn fail_ast(&mut self) {
        self.status = FileStatus::ParseFailed;
    }

    pub fn utir(&self) -> &Utir {
        self.utir.as_ref().unwrap()
    }
    pub fn add_utir(&mut self, utir: Utir) {
        self.utir = Some(utir);
        self.status = FileStatus::UtirSucceed;
    }

    pub fn fail_utir(&mut self) {
        self.status = FileStatus::UtirFailed;
    }

    pub fn fully_qualified_path(&self) -> String {
        self.path
            .with_extension("")
            .into_os_string()
            .into_string()
            .unwrap()
            .replace(std::path::MAIN_SEPARATOR_STR, ".")
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum FileStatus {
    Unloaded,
    Loaded,
    ParseFailed,
    ParseSucceed,
    UtirFailed,
    UtirSucceed,
}

impl<'file> Files<'file> for File {
    type FileId = ();
    type Name = String;
    type Source = &'file str;

    fn name(&'file self, _: Self::FileId) -> Result<Self::Name, codespan_reporting::files::Error> {
        Ok(self
            .pkg
            .borrow()
            .full_path()
            .into_os_string()
            .into_string()
            .unwrap())
    }

    fn source(
        &'file self,
        _: Self::FileId,
    ) -> Result<Self::Source, codespan_reporting::files::Error> {
        Ok(&self.source())
    }

    fn line_index(
        &self,
        _: Self::FileId,
        byte_index: usize,
    ) -> Result<usize, codespan_reporting::files::Error> {
        let line_starts: Vec<usize> =
            codespan_reporting::files::line_starts(self.source()).collect();
        Ok(line_starts
            .binary_search(&byte_index)
            .unwrap_or_else(|next_line| next_line - 1))
    }

    fn line_range(
        &self,
        id: Self::FileId,
        line_index: usize,
    ) -> Result<std::ops::Range<usize>, codespan_reporting::files::Error> {
        fn find_line_start(line_starts: &[usize], line_index: usize) -> Option<usize> {
            use std::cmp::Ordering;

            match line_index.cmp(&line_starts.len()) {
                Ordering::Less => Some(line_starts[line_index]),
                Ordering::Equal => None,
                Ordering::Greater => None,
            }
        }

        let source_len = self.source().len();
        let line_starts: Vec<usize> =
            codespan_reporting::files::line_starts(self.source()).collect();
        let line_start = find_line_start(&line_starts, line_index).unwrap_or(source_len);
        let next_line_start = find_line_start(&line_starts, line_index + 1).unwrap_or(source_len);

        Ok(line_start..next_line_start)
    }
}

impl PartialEq for File {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
    fn ne(&self, other: &Self) -> bool {
        self.path != other.path
    }
}

impl Eq for File {}
