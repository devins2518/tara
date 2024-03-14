use kioku::Arena as KArena;
use std::marker::PhantomData;

pub struct Arena<'arena> {
    inner: KArena,
    _refs: PhantomData<&'arena ()>,
}

impl<'arena> Arena<'arena> {
    pub fn new() -> Self {
        return Self {
            inner: KArena::new(),
            _refs: PhantomData::default(),
        };
    }

    pub fn alloc<'s, T: Copy>(&'s self, val: T) -> &'arena T
    where
        's: 'arena,
    {
        return self.inner.alloc(val);
    }

    pub fn alloc_uninit<'s, T>(&'s self, val: T) -> &'arena T
    where
        's: 'arena,
    {
        return self.inner.alloc_no_copy(val);
    }

    pub fn dupe_str<'s>(&'s self, s: &str) -> &'arena str
    where
        's: 'arena,
    {
        return self.inner.copy_str(s);
    }
}
