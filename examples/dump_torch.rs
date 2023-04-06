#![allow(unused_imports)]
use std::env::args;

use anyhow::{bail, Result};

use repugnant_pickle as rp;

#[cfg(feature = "torch")]
fn main() -> Result<()> {
    let fname = if let Some(fname) = args().nth(1) {
        println!("* Dumping: {fname}\n");
        fname
    } else {
        bail!("Specify pickle filename!");
    };
    let tensors = rp::torch::RepugnantTorchTensors::new_from_file(fname)?;
    println!("{tensors:#?}");
    Ok(())
}

#[cfg(not(feature = "torch"))]
fn main() {
    println!("Compiled without Torch feature.");
}
