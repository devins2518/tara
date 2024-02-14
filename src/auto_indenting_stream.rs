use std::fmt::Write;

pub struct AutoIndentingStream<'a, 'b> {
    underlying: &'b mut std::fmt::Formatter<'a>,
    indent: usize,
    pending_indent: bool,
}

impl<'a, 'b> AutoIndentingStream<'a, 'b> {
    pub fn new(underlying: &'b mut std::fmt::Formatter<'a>) -> Self {
        return Self {
            underlying,
            indent: 0,
            pending_indent: false,
        };
    }
}

impl Write for AutoIndentingStream<'_, '_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        match s {
            "indent+" => {
                self.indent += 4;
                return Ok(());
            }
            "indent-" => {
                self.indent -= 4;
                return Ok(());
            }
            _ => {
                if self.pending_indent {
                    for _ in 0..self.indent {
                        self.underlying.write_char(' ')?;
                    }
                }
                let bytes = s.as_bytes();
                for (i, c) in bytes.iter().enumerate() {
                    self.underlying.write_char(*c as char)?;
                    if (*c == b'\n') && (i != s.len() - 1) {
                        for _ in 0..self.indent {
                            self.underlying.write_char(' ')?;
                        }
                    }
                }
                self.pending_indent = bytes[bytes.len() - 1] == b'\n';
            }
        }
        Ok(())
    }
}
