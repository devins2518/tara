use std::fmt::Display;

#[repr(u32)]
#[derive(Copy, Clone)]
pub enum Mutability {
    Mutable,
    Immutable,
}

impl Display for Mutability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Mutability::Mutable => "mut",
            Mutability::Immutable => "const",
        };
        f.write_str(s)?;
        Ok(())
    }
}

#[repr(u16)]
#[derive(Copy, Clone)]
pub enum Signedness {
    Unsigned,
    Signed,
}

impl Display for Signedness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Signedness::Unsigned => "u",
            Signedness::Signed => "i",
        };
        f.write_str(s)?;
        Ok(())
    }
}
