//! Simple, best-effort scraping of the Python pickle format.
//!
//! See the examples in the `examples/` directory.
//!
//! Something to get you started:
//!
//! ```rust
//! use anyhow::Result;
//! use repugnant_pickle as rp;
//!
//! fn main() -> Result<()> {
//!     let b = b"some bytes of a pickle here";
//!     let (remaining_input, ops) = rp::parse_ops(b)?;
//!     let (values, memo_map) = rp::evaluate(&ops, true)?;
//!     // Use the values here.
//!     Ok(())
//! }
//! ```
//!
//! And here's an example of what the parsed data might
//! look like (from a PyTorch model):
//!
//! ```rust
//! [Build(
//!   Global(
//!     Raw(GLOBAL("collections", "OrderedDict"), [
//!       Seq(Tuple, []),
//!       Seq(Tuple, [
//!         Seq(Dict, [
//!           String("emb.weight"),
//!           Global(Raw(GLOBAL("torch._utils", "_rebuild_tensor_v2")), [
//!             Seq(Tuple, [
//!               PersId(Seq(Tuple, [
//!                 String("storage"),
//!                 Raw(GLOBAL("torch", "BFloat16Storage")),
//!                 String("0"),
//!                 String("cuda:0"),
//!                 Int(430348288),
//!               ])),
//!               Int(327378944),
//!               Seq(Tuple, [Int(1024), Int(50277)]),
//!               Seq(Tuple, [Int(1), Int(1024)]),
//!               Bool(false),
//!               Global(Raw(GLOBAL("collections", "OrderedDict")), [
//!                 Seq(Tuple, [])
//!               ]),
//!             ]),
//!           ]),
//!         ]),
//!         Seq(Dict, [
//!           String("blocks.0.ln1.weight"),
//!           Global(Raw(GLOBAL("torch._utils", "_rebuild_tensor_v2")), [
//!             // Etc.
//!           ]),
//!         ]),
//!       ]),
//!     ]),
//!   ),
//! )]
//! ```

/// Functions used for evaluating Pickle operations.
pub mod eval;

/// Pickle operations.
pub mod ops;

/// Parsers for converting `&[u8]` into a list of
/// Pickle operations.
pub mod parsers;

/// The Value type you can get from evaluating pickle operations.
pub mod value;

#[cfg(feature = "torch")]
pub mod torch;

pub use crate::eval::evaluate;

pub use crate::parsers::parse_ops;

#[cfg(feature = "torch")]
pub use crate::torch::{RepugnantTorchTensor, RepugnantTorchTensors, TensorType};

pub use crate::value::{SequenceType, Value};
