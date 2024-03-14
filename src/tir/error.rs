use crate::{ast::Node, module::file::File};
use anyhow::Result;
use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};

pub enum Failure {
    // Couldn't find root.Top
    CouldNotFindTop,
    // Found root.Top, but couldn't find Top.top
    CouldNotFindTopTop(FoundTop),
    // Found root.Top, but it isn't a module
    TopNotModule,
    // Found Top.top, but it isn't a comb
    TopTopNotComb,
    // TopNotModule(FoundTop),
}

pub struct FoundTop {
    top_label: Label<()>,
}

impl Failure {
    pub fn could_not_find_top() -> Self {
        return Self::CouldNotFindTop;
    }

    pub fn could_not_find_top_top(top: &Node) -> Self {
        let top_label = Label::primary((), top.span);
        return Self::CouldNotFindTopTop(FoundTop { top_label });
    }

    fn labels(&self) -> Vec<Label<()>> {
        match self {
            Self::CouldNotFindTop => vec![],
            Self::CouldNotFindTopTop(top) => {
                vec![top.top_label.clone().with_message("found Top here")]
            }
            Self::TopNotModule => vec![],
            Self::TopTopNotComb => vec![],
        }
    }

    fn notes(&self) -> Vec<String> {
        match self {
            Failure::CouldNotFindTop => vec![],
            Failure::CouldNotFindTopTop { .. } => {
                vec![String::from("Try added a `top` comb to Top")]
            }
            Self::TopNotModule => vec![],
            Self::TopTopNotComb => vec![],
        }
    }

    fn message(&self) -> &'static str {
        match self {
            Failure::CouldNotFindTop => "Could not find root.Top",
            Failure::CouldNotFindTopTop { .. } => "Could not find Top.top()",
            Self::TopNotModule => "Top is not a module",
            Self::TopTopNotComb => "Top.top is not a comb",
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
