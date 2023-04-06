use std::{env::args, fs::File, io::Read};

use anyhow::{bail, Result};

use repugnant_pickle as rp;

fn main() -> Result<()> {
    let mut fp = if let Some(fname) = args().nth(1) {
        println!("* Dumping: {fname}\n");
        File::open(fname)?
    } else {
        bail!("Specify pickle filename!");
    };
    let mut buf = Vec::with_capacity(fp.metadata().map(|md| md.len() as usize).unwrap_or(16384));
    let _ = fp.read_to_end(&mut buf)?;
    match rp::parsers::parse_ops::<nom::error::VerboseError<&[u8]>>(&buf) {
        Ok((_i, ops)) => {
            let (values, _memo) = rp::eval::evaluate(&ops, true)?;
            println!("{values:#?}");
        }
        Err(nom::Err::Error(e) | nom::Err::Failure(e)) => {
            println!("ERROR: {:#?}", e.code);
        }
        _ => (),
    }
    Ok(())
}
