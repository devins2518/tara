use crate::module::file::File;
use anyhow::Result;
use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};

pub enum Failure {
    TopNotFound,
    TopNotModule,
}

impl Failure {
    fn labels(&self) -> Vec<Label<()>> {
        match self {
            Self::TopNotFound => vec![],
            Self::TopNotModule => vec![],
        }
    }

    fn notes(&self) -> Vec<String> {
        match self {
            Self::TopNotFound => vec![],
            Self::TopNotModule => vec![],
        }
    }

    fn message(&self) -> &'static str {
        match self {
            Self::TopNotFound => "Unable to find Top module",
            Self::TopNotModule => "root.Top is not a module!",
        }
    }

    pub fn report<'file, 'comp>(self, file: &'file File<'comp>) -> Result<()> {
        let diagnostic = Diagnostic::error()
            .with_message(self.message())
            .with_labels(self.labels())
            .with_notes(self.notes());

        let writer = StandardStream::stdout(ColorChoice::Always);
        let config = codespan_reporting::term::Config::default();

        term::emit(&mut writer.lock(), &config, file, &diagnostic)?;
        Ok(())
    }
}
