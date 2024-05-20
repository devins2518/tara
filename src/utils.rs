use std::{
    cell::{Ref, RefCell, RefMut},
    fmt::Display,
    hash::Hash,
    mem::MaybeUninit,
    ops::Deref,
    rc::{Rc, Weak},
};

pub mod arena;
pub mod id_arena;
pub mod slice;

// Taken from Calyx https://github.com/calyxir/calyx/
// Copyright 2019 Cornell University Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
// KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
pub struct RRC<T>(Rc<RefCell<T>>);

macro_rules! init_field {
    ($struct:ident, $field:ident, $val:expr) => {{
        use std::ptr::addr_of_mut;
        unsafe {
            addr_of_mut!($struct.$field).write($val);
        }
    }};
}
pub(crate) use init_field;

impl<T> RRC<T> {
    pub fn new(t: T) -> Self {
        Self(Rc::new(RefCell::new(t)))
    }

    pub fn new_uninit() -> (Self, RRC<MaybeUninit<T>>) {
        let rrc = RRC::new(MaybeUninit::uninit());
        (rrc.clone().init(), rrc)
    }

    pub fn borrow(&self) -> Ref<'_, T> {
        self.0.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<'_, T> {
        self.0.borrow_mut()
    }

    pub fn map<R>(&self, f: impl Fn(&T) -> R) -> R {
        let borrow = self.borrow();
        f(&borrow)
    }

    pub fn map_mut<R>(&self, f: impl Fn(&mut T) -> R) -> R {
        let mut borrow = self.borrow_mut();
        f(&mut borrow)
    }
}

impl<T: Clone> RRC<T> {
    pub fn into_inner(&self) -> T {
        let ref_cell: RefCell<T> = (*self.0).clone();
        ref_cell.into_inner()
    }
}

impl<T> RRC<MaybeUninit<T>> {
    pub fn init(self) -> RRC<T> {
        unsafe { std::mem::transmute(self) }
    }
}

impl<T> Deref for RRC<T> {
    type Target = RefCell<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Hash> Hash for RRC<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.borrow().hash(state)
    }
}

impl<T> Clone for RRC<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> From<T> for RRC<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T: Display> Display for RRC<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let borrowed = self.borrow();
        borrowed.fmt(f)
    }
}

impl<T: PartialEq> PartialEq for RRC<T> {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().eq(&*other.borrow())
    }
}

impl<T: Eq> Eq for RRC<T> {}

impl<T: PartialOrd> PartialOrd for RRC<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.borrow().partial_cmp(&*other.borrow())
    }
}

impl<T: Ord> Ord for RRC<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.borrow().cmp(&*other.borrow())
    }
}

/// A Wrapper for a weak RefCell pointer.
/// Used by parent pointers in the internal representation.
pub struct WRC<T>(pub Weak<RefCell<T>>);

impl<T> WRC<T> {
    /// Convinience method to upgrade and extract the underlying internal weak
    /// pointer.
    pub fn upgrade(&self) -> RRC<T> {
        let Some(r) = self.0.upgrade() else {
            unreachable!("weak reference points to a dropped value.");
        };
        RRC(r)
    }
}

/// From implementation with the same signature as `Rc::downgrade`.
impl<T> From<&RRC<T>> for WRC<T> {
    fn from(internal: &RRC<T>) -> Self {
        Self(Rc::downgrade(&internal.0))
    }
}

/// Clone the Weak reference inside the WRC.
impl<T> Clone for WRC<T> {
    fn clone(&self) -> Self {
        Self(Weak::clone(&self.0))
    }
}

impl<T: Hash> Hash for WRC<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.upgrade().hash(state);
    }
}
