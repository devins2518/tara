use codespan::Span;
use std::{
    error::Error as ErrorTrait,
    fmt::{Debug, Display},
};

pub struct Error {
    pub span: Span,
    pub reason: String,
}

impl ErrorTrait for Error {
    fn source(&self) -> Option<&(dyn ErrorTrait + 'static)> {
        None
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}

impl Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

impl Error {
    pub fn new(span: Span, reason: String) -> Self {
        Self { span, reason }
    }
}
