use crate::{ops::*, value::*};

use std::{
    borrow::Cow,
    collections::BTreeMap,
    ops::{Deref, DerefMut},
};

use anyhow::{anyhow, bail, ensure, Ok, Result};

const MAX_DEPTH: usize = 250;
const MAX_PROTOCOL: u8 = 5;

#[derive(Debug, Clone, PartialEq, Default)]
/// Basically just a Vec with some convenience functions.
pub struct PickleStack<'a>(pub Vec<Value<'a>>);

impl<'a> PickleStack<'a> {
    pub fn pop(&mut self) -> Result<Value<'a>> {
        self.0.pop().ok_or_else(|| anyhow!("Stack underrun"))
    }

    pub fn pop_mark(&mut self) -> Result<Vec<Value<'a>>> {
        let markidx = self.find_mark()?;
        let postmark = self.0[markidx + 1..].to_owned();
        self.truncate(markidx);
        Ok(postmark)
    }

    pub fn find_mark(&self) -> Result<usize> {
        Ok(self
            .0
            .iter()
            .enumerate()
            .rfind(|(_idx, op)| matches!(op, Value::Raw(Cow::Borrowed(&PickleOp::MARK))))
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow!("Missing MARK"))?)
    }
}

impl<'a> Deref for PickleStack<'a> {
    type Target = Vec<Value<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> DerefMut for PickleStack<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
/// Basically just a BTreeMap with some convenience functions.
pub struct PickleMemo<'a>(pub BTreeMap<u32, Value<'a>>);

impl<'a> PickleMemo<'a> {
    /// Resolve a Value that could be a reference, but it doesn't look instead
    /// other types. For example if you have `Ref(Ref(Ref(Ref(whatever))))`,
    /// you'll get `whatever` back. But if you have Something(Ref(whatever))
    /// it won't do anything.
    pub fn resolve(&self, mut op: Value<'a>, recursive: bool) -> Result<Value<'a>> {
        let mut count = 0;
        while let Value::Ref(ref mid) = op {
            let val = self.0.get(mid).ok_or_else(|| anyhow!("Bad memo id"))?;
            if !recursive {
                return Ok(val.to_owned());
            }
            op = val.to_owned();
            count += 1;
            if count >= MAX_DEPTH {
                // It be how it be.
                break;
            }
        }

        Ok(op)
    }

    pub fn insert(&mut self, mid: u32, val: Value<'a>) {
        self.0.insert(mid, val);
    }

    /// Like `resolve` but you get a mutable reference.
    pub fn resolve_mut<'b, 'c>(
        &'c mut self,
        op: &'b mut Value<'a>,
        recursive: bool,
    ) -> Result<&'c mut Value<'a>>
    where
        'b: 'c,
        'c: 'b,
    {
        let mut lastmid = if let Value::Ref(ref mid) = op {
            *mid
        } else {
            return Ok(op);
        };

        let mut count = 0;
        while let Value::Ref(mid) = self.0.get(&lastmid).ok_or_else(|| anyhow!("Bad memo id"))? {
            lastmid = *mid;
            if !recursive {
                break;
            }
            count += 1;
            if count >= MAX_DEPTH {
                // We did our best but it fell short.
                break;
            }
        }
        // This unwrap is safe since it's impossible to get here without looking it up
        // non-mut style first.
        Ok(self
            .0
            .get_mut(&lastmid)
            .expect("Impossible: Missing memo id"))
    }

    /// Try to resolve all references in an iterable of
    /// Value. If `fix_values` is true then it will
    /// try to fixup the values.
    pub fn resolve_all_refs_iter(
        &self,
        depth: usize,
        vals: impl IntoIterator<Item = Value<'a>>,
        fix_values: bool,
    ) -> Result<Vec<Value<'a>>> {
        if depth >= MAX_DEPTH {
            // Sometimes things just don't work out the way you hoped.
            return Ok(vals.into_iter().collect::<Vec<_>>());
        }
        vals.into_iter()
            .map(|val| self.resolve_all_refs(depth + 1, val, fix_values))
            .collect::<Result<Vec<_>>>()
    }

    /// Try to resolve all references.
    /// If `fix_values` is true, it will try to fixup
    /// the values.
    pub fn resolve_all_refs(
        &self,
        depth: usize,
        val: Value<'a>,
        fix_values: bool,
    ) -> Result<Value<'a>> {
        if depth >= MAX_DEPTH {
            // It be how it be.
            return Ok(val);
        }
        let rar = |v| self.resolve_all_refs(depth + 1, v, fix_values);
        let rir = |i| self.resolve_all_refs_iter(depth + 1, i, fix_values);

        let output = match val {
            val @ Value::Ref(_) => rar(self.resolve(val, true)?)?,
            Value::App(apped, apps) => Value::App(Box::new(rar(*apped)?), rir(apps)?),
            Value::Object(apped, apps) => Value::Object(Box::new(rar(*apped)?), rir(apps)?),
            Value::Build(apped, apps) => {
                Value::Build(Box::new(rar(*apped)?), Box::new(rar(*apps)?))
            }
            Value::Global(target, args) => Value::Global(Box::new(rar(*target)?), rir(args)?),
            Value::Seq(st, args) => Value::Seq(st, rir(args)?),
            Value::PersId(pid) => Value::PersId(Box::new(rar(*pid)?)),
            val => val,
        };
        if fix_values {
            fix_value(output)
        } else {
            Ok(output)
        }
    }
}

/// Evaluate a slice of pickle ops and try to produce a Vec of
/// Values. You'll also get the memo map back in case you
/// need a way to look up references this crate couldn't handle.
/// You can also pass `resolve_refs` as false and handle
/// the references yourself.
pub fn evaluate<'a>(
    x: &'a [PickleOp],
    resolve_refs: bool,
) -> Result<(Vec<Value<'a>>, PickleMemo<'a>)> {
    let mut stack = PickleStack::default();
    let mut memo = PickleMemo::default();

    fn make_kvlist(items: Vec<Value<'_>>) -> Result<Vec<Value<'_>>> {
        ensure!(items.len() & 1 == 0, "Bad value for setitems");
        let mut kvitems = Vec::with_capacity(items.len());
        let mut it = items.into_iter();
        while let Some(k) = it.next() {
            let v = it.next().expect("Impossible: Missing value item");
            kvitems.push(Value::Seq(SequenceType::Tuple, vec![k, v]));
        }
        Ok(kvitems)
    }

    for op in x.iter() {
        let stack = &mut stack;

        match op {
            PickleOp::MARK => stack.push(Value::Raw(Cow::Borrowed(op))),
            PickleOp::STOP => break,
            PickleOp::POP => {
                let _ = stack.pop()?;
            }
            PickleOp::POP_MARK => {
                let _ = stack.pop_mark()?;
            }
            PickleOp::DUP => {
                let item = stack
                    .last()
                    .ok_or_else(|| anyhow!("Cannot DUP with empty stack"))?
                    .to_owned();
                stack.push(item);
            }
            PickleOp::PERSID(pid) => stack.push(Value::PersId(Box::new(Value::String(pid)))),
            PickleOp::BINPERSID => {
                let pid = stack.pop()?;
                stack.push(Value::PersId(Box::new(pid)));
            }
            PickleOp::REDUCE => {
                let args = memo.resolve(stack.pop()?, true)?;
                let target = memo.resolve(stack.pop()?, true)?;
                stack.push(Value::Global(Box::new(target), vec![args]));
            }
            PickleOp::BUILD => {
                let args = Box::new(memo.resolve(stack.pop()?, true)?);
                let target = Box::new(memo.resolve(stack.pop()?, true)?);
                stack.push(Value::Build(target, args));
            }
            PickleOp::EMPTY_DICT => stack.push(Value::Seq(SequenceType::Dict, Default::default())),
            PickleOp::GET(mids) => stack.push(Value::Ref(mids.parse()?)),
            PickleOp::BINGET(mid) => stack.push(Value::Ref(*mid as u32)),
            PickleOp::LONG_BINGET(mid) => stack.push(Value::Ref(*mid)),
            PickleOp::EMPTY_LIST => stack.push(Value::Seq(SequenceType::List, Default::default())),
            PickleOp::BINPUT(mid) => {
                let mid = *mid as u32;
                memo.insert(mid, stack.pop()?);
                stack.push(Value::Ref(mid));
            }
            PickleOp::LONG_BINPUT(mid) => {
                memo.insert(*mid, stack.pop()?);
                stack.push(Value::Ref(*mid));
            }
            PickleOp::TUPLE => {
                let postmark = stack.pop_mark()?;
                stack.push(Value::Seq(SequenceType::Tuple, postmark));
            }
            PickleOp::EMPTY_TUPLE => {
                stack.push(Value::Seq(SequenceType::Tuple, Default::default()))
            }
            PickleOp::SETITEM => {
                let v = stack.pop()?;
                let k = stack.pop()?;
                let top = stack
                    .last_mut()
                    .ok_or_else(|| anyhow!("Unexpected empty stack"))?;
                let rtop = memo.resolve_mut(top, true)?;
                match rtop {
                    Value::Global(_, args) | Value::Seq(_, args) => {
                        args.push(Value::Seq(SequenceType::Tuple, vec![k, v]));
                    }
                    _wut => bail!("Bad stack top for SETITEM!"),
                }
            }
            PickleOp::SETITEMS => {
                let kvitems = make_kvlist(stack.pop_mark()?)?;
                let top = stack
                    .last_mut()
                    .ok_or_else(|| anyhow!("Unexpected empty stack"))?;
                let rtop = memo.resolve_mut(top, true)?;
                match rtop {
                    Value::Global(_, args) | Value::Seq(_, args) => {
                        args.push(Value::Seq(SequenceType::Tuple, kvitems));
                    }
                    _wut => bail!("Bad stack top for SETITEMS"),
                }
            }
            PickleOp::PROTO(proto) => {
                ensure!(*proto <= MAX_PROTOCOL, "Unsupported protocol {proto}")
            }
            PickleOp::TUPLE1 => {
                let t1 = stack.pop()?;
                stack.push(Value::Seq(SequenceType::Tuple, vec![t1]));
            }
            PickleOp::TUPLE2 => {
                let (t1, t2) = (stack.pop()?, stack.pop()?);
                stack.push(Value::Seq(SequenceType::Tuple, vec![t1, t2]));
            }
            PickleOp::TUPLE3 => {
                let (t1, t2, t3) = (stack.pop()?, stack.pop()?, stack.pop()?);
                stack.push(Value::Seq(SequenceType::Tuple, vec![t1, t2, t3]));
            }
            PickleOp::APPEND => {
                let v = stack.pop()?;
                let top = stack
                    .last_mut()
                    .ok_or_else(|| anyhow!("Unexpected empty stack"))?;
                let rtop = memo.resolve_mut(top, true)?;
                match rtop {
                    Value::Global(_, args) | Value::Seq(_, args) => {
                        args.push(v);
                    }
                    _wut => bail!("Bad stack top for APPEND!"),
                }
            }
            PickleOp::APPENDS => {
                let postmark = stack.pop_mark()?;
                let top = stack
                    .last_mut()
                    .ok_or_else(|| anyhow!("Unexpected empty stack"))?;
                let rtop = memo.resolve_mut(top, true)?;

                match rtop {
                    Value::Global(_, args) | Value::Seq(_, args) => {
                        args.extend(postmark);
                    }
                    _wut => bail!("Bad stack top for APPENDS"),
                }
            }
            PickleOp::DICT => {
                let kvitems = make_kvlist(stack.pop_mark()?)?;
                stack.push(Value::Seq(SequenceType::Dict, kvitems));
            }
            PickleOp::LIST => {
                let items = stack.pop_mark()?;
                stack.push(Value::Seq(SequenceType::List, items));
            }
            PickleOp::INST(mn, cn) => {
                let args = stack.pop_mark()?;
                stack.push(Value::Object(
                    Box::new(Value::Seq(
                        SequenceType::Tuple,
                        vec![Value::String(mn), Value::String(cn)],
                    )),
                    args,
                ))
            }
            PickleOp::OBJ => {
                let markidx = stack.find_mark()?;
                let args = stack.0[markidx + 2..].to_owned();
                let cls = stack.0[markidx + 1].clone();
                stack.0.truncate(markidx);
                stack.push(Value::Object(Box::new(cls), args));
            }
            PickleOp::PUT(midstr) => {
                // Note: This technically incorrect since the memo id could actually be a string, but
                // it doesn't seem like that happens in practice.
                let mid = midstr.parse()?;
                memo.insert(mid, stack.pop()?);
                stack.push(Value::Ref(mid));
            }
            PickleOp::NEWOBJ => {
                let (args, cls) = (stack.pop()?, stack.pop()?);
                stack.push(Value::Object(Box::new(cls), vec![args]))
            }
            PickleOp::EMPTY_SET => stack.push(Value::Seq(SequenceType::Set, vec![])),
            PickleOp::ADDITEMS => {
                let postmark = stack.pop_mark()?;
                let top = stack
                    .last_mut()
                    .ok_or_else(|| anyhow!("Unexpected empty stack"))?;
                let rtop = memo.resolve_mut(top, true)?;

                match rtop {
                    Value::Global(_, args) | Value::Seq(_, args) => {
                        args.extend(postmark);
                    }
                    _wut => bail!("Bad stack top for ADDITEMS"),
                }
            }
            PickleOp::FROZENSET => {
                let items = stack.pop_mark()?;
                stack.push(Value::Seq(SequenceType::FrozenSet, items));
            }
            PickleOp::NEWOBJ_EX => {
                let (kwargs, args, cls) = (stack.pop()?, stack.pop()?, stack.pop()?);
                stack.push(Value::Object(
                    Box::new(cls),
                    vec![Value::Seq(SequenceType::Tuple, vec![args, kwargs])],
                ))
            }
            PickleOp::STACK_GLOBAL => {
                let (gn, mn) = (
                    memo.resolve(stack.pop()?, true)?,
                    memo.resolve(stack.pop()?, true)?,
                );
                stack.push(Value::Global(
                    Box::new(Value::Seq(SequenceType::Tuple, vec![gn, mn])),
                    vec![],
                ));
            }
            PickleOp::MEMOIZE => {
                let item = stack.last().ok_or_else(|| anyhow!("Stack underrun"))?;
                memo.insert(memo.0.len() as u32, item.to_owned());
            }

            // Fallthrough case is just to push the op onto the stack as a Value::Raw.
            op => stack.push(Value::Raw(Cow::Borrowed(op))),
        }
    }
    if !resolve_refs {
        return Ok((stack.0, memo));
    }
    let stack = memo.resolve_all_refs_iter(0, stack.0, true)?;

    Ok((stack, memo))
}
