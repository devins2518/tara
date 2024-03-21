use std::cell::RefCell;
use std::rc::Rc;

pub mod arena;
pub mod id_arena;
pub mod slice;
pub type RRC<T> = Rc<RefCell<T>>;
