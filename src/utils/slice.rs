use std::fmt::Display;
use std::ops::Deref;
use std::path::Path;
use std::pin::Pin;

pub struct OwnedSlice<T: Unpin> {
    slice: Pin<Box<[T]>>,
}

impl<T: Unpin> From<Vec<T>> for OwnedSlice<T> {
    fn from(value: Vec<T>) -> Self {
        return Self {
            slice: Pin::new(value.into_boxed_slice()),
        };
    }
}

impl<T: Unpin> Deref for OwnedSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        return self.slice.as_ref().get_ref();
    }
}

impl<T: Unpin> AsRef<[T]> for OwnedSlice<T> {
    fn as_ref(&self) -> &[T] {
        return self.deref();
    }
}

#[derive(Clone)]
pub struct OwnedString {
    slice: Pin<Box<str>>,
}

impl From<String> for OwnedString {
    fn from(value: String) -> Self {
        return Self {
            slice: Pin::new(value.into_boxed_str()),
        };
    }
}

impl Deref for OwnedString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        return self.slice.as_ref().get_ref();
    }
}

impl Display for OwnedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&*self)?;
        return Ok(());
    }
}

impl AsRef<str> for OwnedString {
    fn as_ref(&self) -> &str {
        return self.deref();
    }
}

impl AsRef<Path> for OwnedString {
    fn as_ref(&self) -> &Path {
        return Path::new(self.deref());
    }
}
