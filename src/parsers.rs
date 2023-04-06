use std::str::Utf8Error;

use nom::{
    bytes::complete::*, combinator::*, error as ne, multi::*, number::complete::*, sequence::tuple,
    *,
};

use crate::ops::*;

fn parse_string_nl<'a, E>(i: &'a [u8]) -> IResult<&'a [u8], &'a str, E>
where
    E: ne::ParseError<&'a [u8]> + ne::FromExternalError<&'a [u8], Utf8Error>,
{
    let (i, s) = map_res(take_till(|c| c == b'\n'), std::str::from_utf8)(i)?;
    let (i, _) = tag(b"\n")(i)?;
    IResult::Ok((i, s))
}

/// Parse 1+ ops into a Vec. It's a nom parser.
pub fn parse_ops<'a, E>(i: &'a [u8]) -> IResult<&'a [u8], Vec<PickleOp>>
where
    E: ne::ParseError<&'a [u8]> + ne::FromExternalError<&'a [u8], Utf8Error>,
{
    many1(map(parse_op::<E>, |op| op))(i)
}

/// Parse a single op. It's nom parser.
pub fn parse_op<'a, E>(i: &'a [u8]) -> IResult<&'a [u8], PickleOp>
where
    E: ne::ParseError<&'a [u8]> + ne::FromExternalError<&'a [u8], Utf8Error>,
{
    let (i, opcode) = u8(i)?;
    IResult::Ok((
        i,
        match opcode {
            p_op::MARK => PickleOp::MARK,
            p_op::STOP => PickleOp::STOP,
            p_op::POP => PickleOp::POP,
            p_op::POP_MARK => PickleOp::POP_MARK,
            p_op::DUP => PickleOp::DUP,
            p_op::FLOAT => return map(parse_string_nl, PickleOp::FLOAT)(i),
            p_op::INT => return map(parse_string_nl, PickleOp::INT)(i),
            p_op::BININT => return map(le_i32, PickleOp::BININT)(i),
            p_op::BININT1 => return map(u8, PickleOp::BININT1)(i),
            p_op::LONG => return map(parse_string_nl, PickleOp::LONG)(i),
            p_op::BININT2 => return map(le_u16, PickleOp::BININT2)(i),
            p_op::NONE => PickleOp::NONE,
            p_op::PERSID => return map(parse_string_nl, PickleOp::PERSID)(i),
            p_op::BINPERSID => PickleOp::BINPERSID,
            p_op::REDUCE => PickleOp::REDUCE,
            p_op::STRING => return map(parse_string_nl, PickleOp::STRING)(i),
            p_op::BINSTRING => return map(length_data(le_u32), PickleOp::BINSTRING)(i),
            p_op::SHORT_BINSTRING => return map(length_data(u8), PickleOp::SHORT_BINSTRING)(i),
            p_op::UNICODE => return map(parse_string_nl, PickleOp::UNICODE)(i),
            p_op::BINUNICODE => {
                return map(
                    map_res(length_data(le_u32), std::str::from_utf8),
                    PickleOp::BINUNICODE,
                )(i)
            }
            p_op::APPEND => PickleOp::APPEND,
            p_op::BUILD => PickleOp::BUILD,
            p_op::GLOBAL => {
                return map(tuple((parse_string_nl, parse_string_nl)), |(mn, gn)| {
                    PickleOp::GLOBAL(mn, gn)
                })(i);
            }
            p_op::DICT => PickleOp::DICT,
            p_op::EMPTY_DICT => PickleOp::EMPTY_DICT,
            p_op::APPENDS => PickleOp::APPENDS,
            p_op::GET => return map(parse_string_nl, PickleOp::GET)(i),
            p_op::BINGET => return map(u8, PickleOp::BINGET)(i),
            p_op::INST => {
                return map(tuple((parse_string_nl, parse_string_nl)), |(mn, cn)| {
                    PickleOp::INST(mn, cn)
                })(i);
            }
            p_op::LONG_BINGET => return map(le_u32, PickleOp::LONG_BINGET)(i),
            p_op::LIST => PickleOp::LIST,
            p_op::EMPTY_LIST => PickleOp::EMPTY_LIST,
            p_op::OBJ => PickleOp::OBJ,
            p_op::PUT => return map(parse_string_nl, PickleOp::PUT)(i),
            p_op::BINPUT => return map(u8, PickleOp::BINPUT)(i),
            p_op::LONG_BINPUT => return map(le_u32, PickleOp::LONG_BINPUT)(i),
            p_op::SETITEM => PickleOp::SETITEM,
            p_op::TUPLE => PickleOp::TUPLE,
            p_op::EMPTY_TUPLE => PickleOp::EMPTY_TUPLE,
            p_op::SETITEMS => PickleOp::SETITEMS,
            p_op::BINFLOAT => return map(le_f64, PickleOp::BINFLOAT)(i),
            p_op::PROTO => return map(u8, PickleOp::PROTO)(i),
            p_op::NEWOBJ => PickleOp::NEWOBJ,
            p_op::EXT1 => return map(u8, PickleOp::EXT1)(i),
            p_op::EXT2 => return map(le_i16, PickleOp::EXT2)(i),
            p_op::EXT4 => return map(le_i32, PickleOp::EXT4)(i),
            p_op::TUPLE1 => PickleOp::TUPLE1,
            p_op::TUPLE2 => PickleOp::TUPLE2,
            p_op::TUPLE3 => PickleOp::TUPLE3,
            p_op::NEWTRUE => PickleOp::NEWTRUE,
            p_op::NEWFALSE => PickleOp::NEWFALSE,
            p_op::LONG1 => return map(length_data(u8), PickleOp::LONG1)(i),
            p_op::LONG4 => return map(length_data(le_u32), PickleOp::LONG4)(i),
            p_op::BINBYTES => return map(length_data(le_u32), PickleOp::BINBYTES)(i),
            p_op::BINBYTES8 => return map(length_data(le_u64), PickleOp::BINBYTES8)(i),
            p_op::SHORT_BINBYTES => return map(length_data(u8), PickleOp::SHORT_BINBYTES)(i),
            p_op::BINUNICODE8 => {
                return map(
                    map_res(length_data(le_u64), std::str::from_utf8),
                    PickleOp::BINUNICODE8,
                )(i)
            }
            p_op::SHORT_BINUNICODE => {
                return map(
                    map_res(length_data(u8), std::str::from_utf8),
                    PickleOp::BINUNICODE8,
                )(i)
            }
            p_op::EMPTY_SET => PickleOp::EMPTY_SET,
            p_op::ADDITEMS => PickleOp::ADDITEMS,
            p_op::FROZENSET => PickleOp::FROZENSET,
            p_op::NEWOBJ_EX => PickleOp::NEWOBJ_EX,
            p_op::STACK_GLOBAL => PickleOp::STACK_GLOBAL,
            p_op::MEMOIZE => PickleOp::MEMOIZE,
            p_op::FRAME => return map(le_u64, PickleOp::FRAME)(i),
            p_op::BYTEARRAY8 => return map(length_data(le_u64), PickleOp::BYTEARRAY8)(i),
            p_op::NEXT_BUFFER => PickleOp::NEXT_BUFFER,
            p_op::READONLY_BUFFER => PickleOp::READONLY_BUFFER,
            _ => return cut(nom::error::context("Bad opcode", fail))(i),
        },
    ))
}
