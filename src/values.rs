#[derive(PartialEq, Eq, Hash)]
pub enum Value<'a> {
    // Raw bytes that can be interpreted in any way
    Bytes(&'a [u8]),
    // An aggregate value that is laid out according to how a type dictates
    Aggregate(&'a [Value<'a>]),
}
