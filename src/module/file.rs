use crate::{ast::Ast, module::decls::Decl, utir::Utir};
use codespan_reporting::files::Files;

pub struct File<'comp> {
    path: &'comp str,
    status: FileStatus,
    source: Option<&'comp str>,
    ast: Option<&'comp Ast>,
    utir: Option<&'comp Utir<'comp>>,
    pub root_decl: Option<&'comp Decl<'comp>>,
}

impl<'comp> File<'comp> {
    pub fn new(path: &'comp str) -> Self {
        return Self {
            path,
            status: FileStatus::Unloaded,
            source: None,
            ast: None,
            utir: None,
            root_decl: None,
        };
    }

    // Asserts that source is loaded
    pub fn source<'source>(&self) -> &'source str
    where
        'comp: 'source,
    {
        self.source.unwrap()
    }
    pub fn add_source(&mut self, source: &'comp str) {
        self.source = Some(source);
        self.status = FileStatus::Loaded;
    }

    pub fn ast(&self) -> &Ast {
        self.ast.unwrap()
    }
    pub fn add_ast(&mut self, ast: &'comp Ast) {
        self.ast = Some(ast);
        self.status = FileStatus::ParseSucceed;
    }

    pub fn fail_ast(&mut self) {
        self.status = FileStatus::ParseFailed;
    }

    pub fn utir(&self) -> &'comp Utir<'comp> {
        self.utir.unwrap()
    }
    pub fn add_utir(&mut self, utir: &'comp Utir<'comp>) {
        self.utir = Some(utir);
        self.status = FileStatus::UtirSucceed;
    }

    pub fn fail_utir(&mut self) {
        self.status = FileStatus::UtirFailed;
    }
}

enum FileStatus {
    Unloaded,
    Loaded,
    ParseFailed,
    ParseSucceed,
    UtirFailed,
    UtirSucceed,
}

impl<'comp, 'file> Files<'file> for File<'comp>
where
    'comp: 'file,
{
    type FileId = ();
    type Name = &'file str;
    type Source = &'file str;

    fn name(&self, _: Self::FileId) -> Result<Self::Name, codespan_reporting::files::Error> {
        Ok(self.path)
    }

    fn source(&self, _: Self::FileId) -> Result<Self::Source, codespan_reporting::files::Error> {
        Ok(self.source())
    }

    fn line_index(
        &self,
        id: Self::FileId,
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
