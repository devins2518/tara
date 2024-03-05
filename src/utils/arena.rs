use std::{
    cell::UnsafeCell,
    cmp::{Eq, PartialEq},
    hash::Hash,
    marker::PhantomData,
    ops::Add,
};

pub struct Arena<T> {
    inner: UnsafeCell<Vec<T>>,
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        return Self {
            inner: UnsafeCell::new(Vec::new()),
        };
    }

    fn get_self(&self) -> &mut Self {
        let const_ptr = self as *const Self;
        let mut_ptr = const_ptr as *mut Self;
        let mut_self = unsafe { &mut *mut_ptr };
        return mut_self;
    }

    fn get_inner(&self) -> &mut Vec<T> {
        let mut_self = self.get_self();
        let vec = mut_self.inner.get_mut();
        return vec;
    }

    pub fn alloc(&self, val: T) -> Id<T> {
        let vec = self.get_inner();
        let id = (vec.len() as u32).into();
        vec.push(val);
        return id;
    }

    pub fn reserve(&self) -> Id<T> {
        let vec = self.get_inner();
        let id = (vec.len() as u32).into();
        let val = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        vec.push(val);
        return id;
    }

    pub fn set(&self, id: Id<T>, val: T) {
        let vec = self.get_inner();
        let idx: u32 = id.into();
        vec[idx as usize] = val;
    }

    pub fn len(&self) -> usize {
        let vec = self.get_inner();
        return vec.len();
    }

    pub fn get(&self, id: Id<T>) -> T
    where
        T: Copy,
    {
        return self.get_inner()[u32::from(id) as usize];
    }
}

pub trait ExtraArenaContainable<const N: usize>: From<[u32; N]> + Into<[u32; N]> {}

impl Arena<u32> {
    pub fn insert_extra<const N: usize, T: ExtraArenaContainable<N>>(&self, val: T) -> Id<T> {
        let slice: [u32; N] = val.into();
        let ret = self.reserve();
        for b in &slice[1..] {
            let _ = self.alloc(*b);
        }
        self.set(ret, slice[0]);
        return ret.from_u32();
    }

    pub fn get_extra<const N: usize, T: ExtraArenaContainable<N>>(&self, id: Id<u32>) -> T {
        let idx: usize = u32::from(id) as usize;
        let slice = self.get_inner()[idx..(idx + N)].try_into().unwrap();
        return T::from(slice);
    }

    pub fn set_extra<const N: usize, T: ExtraArenaContainable<N>>(&self, id: Id<T>, val: T) {
        let idx: usize = u32::from(id) as usize;
        let u32s = val.into();
        for i in 0..N {
            self.get_inner()[idx + i] = u32s[i];
        }
    }

    pub fn slice<const N: usize, T: ExtraArenaContainable<N>>(
        &self,
        start: Id<T>,
        len: u32,
    ) -> &[T] {
        let vec = self.get_inner();
        let start: u32 = start.into();
        let end: u32 = start + (len * N as u32);
        let raw_slice = &vec[start as usize..end as usize];
        return unsafe {
            core::slice::from_raw_parts(raw_slice.as_ptr() as *const T, len as usize)
        };
    }
}

pub struct ArenaRef<T: Copy> {
    pub(super) data: Box<[T]>,
}

impl<T: Copy> From<Arena<T>> for ArenaRef<T> {
    fn from(value: Arena<T>) -> Self {
        let data = value.inner.into_inner().into_boxed_slice();
        return Self { data };
    }
}

impl<T: Copy> ArenaRef<T> {
    pub fn iter(&mut self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
}

pub const ID_U32S: usize = 1;
pub struct Id<T> {
    id: u32,
    _phantom: PhantomData<T>,
}

impl<T> Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        return self.id == other.id;
    }
}

impl<T> Eq for Id<T> {}

impl<T> ExtraArenaContainable<ID_U32S> for Id<T> {}
impl<T> From<[u32; ID_U32S]> for Id<T> {
    fn from(value: [u32; ID_U32S]) -> Self {
        return Self::from(value[0]);
    }
}

impl<T> From<Id<T>> for [u32; ID_U32S] {
    fn from(value: Id<T>) -> Self {
        return [u32::from(value)];
    }
}

impl Id<u32> {
    pub fn from_u32<U>(&self) -> Id<U> {
        return Id::from(self.id);
    }
}

impl<T> Id<T> {
    pub fn to_u32(&self) -> Id<u32> {
        return Id::from(self.id);
    }
}

impl<T> Copy for Id<T> {}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        return Self::from(self.id);
    }
}

impl<T> From<u32> for Id<T> {
    fn from(id: u32) -> Self {
        return Self {
            id,
            _phantom: PhantomData,
        };
    }
}

impl<T> From<Id<T>> for u32 {
    fn from(id: Id<T>) -> u32 {
        return id.id;
    }
}

impl<T> Add<u32> for Id<T> {
    type Output = Id<T>;

    fn add(self, rhs: u32) -> Self::Output {
        return Self::from(self.id + rhs);
    }
}

impl<T> Add<usize> for Id<T> {
    type Output = Id<T>;

    fn add(self, rhs: usize) -> Self::Output {
        return Self::from(self.id + rhs as u32);
    }
}
