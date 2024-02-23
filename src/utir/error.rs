use crate::ast::{Ast, Node};
use anyhow::Result;
use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::term;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};

pub enum Failure {
    VariableShadowing(Shadow),
    UnknownIdentifier(UnknownDecl),
}

pub struct Shadow {
    shadow: Label<()>,
    original: Label<()>,
}

pub struct UnknownDecl {
    reference: Label<()>,
}

impl Failure {
    pub fn shadow(shadow: &Node, original: &Node) -> Self {
        let shadow_label = Label::primary((), shadow.span);
        let original_label = Label::secondary((), original.span);
        return Self::VariableShadowing(Shadow {
            original: original_label,
            shadow: shadow_label,
        });
    }

    pub fn unknown(node: &Node) -> Self {
        let node_label = Label::primary((), node.span);
        return Self::UnknownIdentifier(UnknownDecl {
            reference: node_label,
        });
    }

    fn labels(&self) -> Vec<Label<()>> {
        match self {
            Self::VariableShadowing(shadow) => vec![
                shadow.shadow.clone().with_message("shadow declared here"),
                shadow
                    .original
                    .clone()
                    .with_message("original declared here"),
            ],
            Self::UnknownIdentifier(unknown) => vec![unknown
                .reference
                .clone()
                .with_message("unknown identifier used here")],
        }
    }

    fn notes(&self) -> Vec<String> {
        match self {
            Self::VariableShadowing(_) => vec![],
            Self::UnknownIdentifier(_) => vec![],
        }
    }

    fn message(&self) -> &'static str {
        match self {
            Self::VariableShadowing(_) => "Variable declaration shadows previous declaration",
            Self::UnknownIdentifier(_) => "Use of unknown identifier",
        }
    }

    pub fn report(self, ast: &Ast) -> Result<()> {
        let diagnostic = Diagnostic::error()
            .with_message(self.message())
            .with_labels(self.labels())
            .with_notes(self.notes());

        let writer = StandardStream::stdout(ColorChoice::Always);
        let config = codespan_reporting::term::Config::default();

        term::emit(&mut writer.lock(), &config, ast.source, &diagnostic)?;
        Ok(())
    }
}
