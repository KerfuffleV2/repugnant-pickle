use std::borrow::Cow;

use anyhow::Result;
use num_bigint::BigInt;

use crate::ops::PickleOp;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
/// The types of sequences that exist.
pub enum SequenceType {
    List,
    Dict,
    Tuple,
    Set,
    FrozenSet,
}

#[derive(Debug, Clone, PartialEq)]
/// A processed value.
pub enum Value<'a> {
    /// Types that we can't handle or just had to give up on processing.
    Raw(Cow<'a, PickleOp<'a>>),

    /// A reference. You might be able to look it up in the memo map
    /// unless there's something weird going on like recursive references.
    /// You generally shouldn't see this in the result unless bad things
    /// are going on...
    Ref(u32),

    /// The result of applying a thing to another thing. We're not
    /// Python so we don't really know what a "thing" is.
    App(Box<Value<'a>>, Vec<Value<'a>>),

    /// An object or something. The first tuple member is the
    /// thing, the second one is the arguments it got applied to.
    Object(Box<Value<'a>>, Vec<Value<'a>>),

    /// Something we tried to build. The first tuple member is the
    /// thing, the second one is the arguments it got applied to.
    Build(Box<Value<'a>>, Box<Value<'a>>),

    /// References to persistant storage. They basically could be anything.
    /// You kind of have to know what the thing you're trying to
    /// interface wants to use as keys for persistant storage.
    /// Good luck.
    PersId(Box<Value<'a>>),

    /// A global value of some kind. The first tuple member is
    /// the thing, the second one is the arguments it got applied to.
    Global(Box<Value<'a>>, Vec<Value<'a>>),

    /// A sequence. We don't really distinguish between them
    /// much. The one exception is when the SequenceType is
    /// Dict we try to split the flat list of `[k, v, k, v, k, v]`
    /// into a list of tuples with the key and value.
    Seq(SequenceType, Vec<Value<'a>>),

    /// A string, but not the crazy strings that have to be
    /// unescaped as if they were Python strings. If you
    /// need one of those, look for it inside a `Value::Raw`.
    String(&'a str),

    /// Some bytes. It might be a byte array or a binary
    /// string that couldn't get UTF8 decoded. We do the best
    /// we can.
    Bytes(&'a [u8]),

    /// An integer, but not the crazy kind that comes as a string
    /// that has to be parsed. You can look in `Value::RawNum` for
    /// those.
    Int(i64),

    /// An integer that can't fit in i64.
    BigInt(BigInt),

    /// An float, but not the crazy kind that comes as a string
    /// that has to be parsed. You can look in `Value::RawNum` for
    /// those.
    Float(f64),

    /// Some kind of weird number we can't handle.
    RawNum(PickleOp<'a>),

    /// A boolean value.
    Bool(bool),

    /// Python `None`.
    None,
}

/// Attempt to fix up a value from `Value::Raw(...)` into something
/// more reasonable.
pub fn fix_value(val: Value<'_>) -> Result<Value<'_>> {
    use once_cell::sync::Lazy;
    static BI64MIN: Lazy<BigInt> = Lazy::new(|| BigInt::from(i64::MIN));
    static BI64MAX: Lazy<BigInt> = Lazy::new(|| BigInt::from(i64::MAX));
    match val {
        Value::Raw(ref rv) => Ok(match rv.as_ref() {
            PickleOp::BININT(val) => Value::Int(*val as i64),
            PickleOp::BININT1(val) => Value::Int(*val as i64),
            PickleOp::BININT2(val) => Value::Int(*val as i64),
            PickleOp::LONG1(b) | PickleOp::LONG4(b) if !b.is_empty() => {
                let blen = b.len();
                let is_neg = b[blen - 1] & 80 != 0;
                let mut bint = BigInt::from_bytes_le(num_bigint::Sign::Plus, b);
                if is_neg {
                    bint -= BigInt::from(1) << (blen * 8);
                }
                if &bint >= &BI64MIN && &bint <= &BI64MAX {
                    (&bint)
                        .try_into()
                        .map_or_else(|_| Value::BigInt(bint), Value::Int)
                } else {
                    Value::BigInt(bint)
                }
            }
            PickleOp::BINFLOAT(val) => Value::Float(*val),
            PickleOp::BINUNICODE(s) | PickleOp::BINUNICODE8(s) | PickleOp::SHORT_BINUNICODE(s) => {
                Value::String(s)
            }
            PickleOp::BINBYTES(b)
            | PickleOp::BINBYTES8(b)
            | PickleOp::SHORT_BINBYTES(b)
            | PickleOp::BYTEARRAY8(b) => Value::Bytes(b),
            // This isn't how Pickle actually works but we just try to UTF8 decode the
            // string and if it fails, we make it a bytes value instead. If anyone
            // actually cares they can just fix values themselves or recover the raw bytes
            // from the UTF8 string (it's guaranteed to be reversible, as far as I know).
            PickleOp::BINSTRING(b) | PickleOp::SHORT_BINSTRING(b) => std::str::from_utf8(b)
                .map(Value::String)
                .unwrap_or_else(|_| Value::Bytes(b)),
            PickleOp::NEWTRUE => Value::Bool(true),
            PickleOp::NEWFALSE => Value::Bool(false),
            PickleOp::NONE => Value::None,
            PickleOp::INT("01") => Value::Bool(true),
            PickleOp::INT("00") => Value::Bool(false),
            PickleOp::INT(_)
            | PickleOp::FLOAT(_)
            | PickleOp::LONG(_)
            | PickleOp::LONG1(_)
            | PickleOp::LONG4(_) => Value::RawNum(rv.clone().into_owned()),
            _ => val,
        }),
        val => Ok(val),
    }
}
