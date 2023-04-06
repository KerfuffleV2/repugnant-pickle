# repugnant-pickle

*Because it is, isn't it?*

This is a Rust crate for dealing with the Python pickle format.

It also has support for opening PyTorch files (`.pth`, `.pt`) which
are basically ZIP files with a pickle hidden inside.

## What?

The Python pickle format. There are people who get paid way more than me who
spent massive amounts of effort working on copy protection schemes that were
less effective at preventing interoperation than the Python pickle format.

Why would anyone actually make such a thing? My theory is that the originator
has a deep and abiding hate for their fellow humans. The only thing that gives
them any pleasure is to cause pain to as many people as possible. To that end:
the Python pickle format.

It's basically impossible to handle reliably unless you're Python. In an ideal
world, you'd just leave the pickles alone but sadly that is not the world we
find ourselves living in. Sometimes you need to get some data out of a pickle
file and you don't want to embed a whole Python interpreter into your application
or something crazy like that.

That's where `repugnant-pickle` comes in. It does what it can and gives you what
it came up with.

The name is a riff on BeautifulSoup, a HTML scraper library. It didn't (doesn't?) try
to parse HTML completely (which is insanely hard), it just gives you a relatively simple
way to to to extract the data you need. Most of the time that's good enough.


## Example

Here's the kind of output you could expect to get from parsing a
PyTorch file:

```rust
[Build(
  Global(
    Raw(GLOBAL("collections", "OrderedDict"), [
      Seq(Tuple, []),
      Seq(Tuple, [
        Seq(Dict, [
          String("emb.weight"),
          Global(Raw(GLOBAL("torch._utils", "_rebuild_tensor_v2")), [
            Seq(Tuple, [
              PersId(Seq(Tuple, [
                String("storage"),
                Raw(GLOBAL("torch", "BFloat16Storage")),
                String("0"),
                String("cuda:0"),
                Int(430348288),
              ])),
              Int(327378944),
              Seq(Tuple, [Int(50277), Int(1024)]),
              Seq(Tuple, [Int(1024), Int(1)]),
              Bool(false),
              Global(Raw(GLOBAL("collections", "OrderedDict")), [
                Seq(Tuple, [])
              ]),
            ]),
          ]),
        ]),
        Seq(Dict, [
          String("blocks.0.ln1.weight"),
          Global(Raw(GLOBAL("torch._utils", "_rebuild_tensor_v2")), [
            // Etc
          ]),
        ]),
      ]),
    ]),
  ),
)]
```

## What about `serde-pickle`?

Use it if you can. It absolutely will be a much nicer experience than this crate.
However, there are things it can't handle such as persistant IDs. PyTorch files
use persistant IDs, so you don't really have a choice in that case.

There are also some odd choices like forcing the Rust toolchain to version 1.41.

## PyTorch

You can enable support for _attempting_ to deal with PyTorch files with the
`torch` feature. If your Torch file has weird stuff, you may have to deal with
it manually. Look in [`src/torch.rs`](src/torch.rs) for an example of where to start.

Otherwise, you can use
`repugnant_pickle::torch::RepugnantTorchTensors::new_from_file`
to load the tensor metadata. For example, if it works, you'll get something like
this:

```Rust
RepugnantTorchTensors(
   [
       RepugnantTorchTensor {
           name: "emb.weight",
           device: "cuda:0",
           tensor_type: BFloat16,
           storage: "archive/data/0",
           storage_len: 430348288,
           storage_offset: 327378944,
           absolute_offset: 327445248,
           shape: [50277, 1024],
           stride: [1024, 1],
           requires_grad: false,
       },
       RepugnantTorchTensor {
           name: "blocks.0.ln1.weight",
           device: "cuda:0",
           tensor_type: BFloat16,
           storage: "archive/data/0",
           storage_len: 430348288,
           storage_offset: 13639680,
           absolute_offset: 13705984,
           shape: [1024],
           stride: [1],
           requires_grad: false,
       },
   ]
```

You'll have to calculate the length yourself from the
type and shape. For example, the first tensor has shape
`[1024, 50277]` and type `BFloat16`. So the length in
bytes would be `(1024 * 50277) * 2` and the data would
start at offset `327445248` in the Torch file, or
if you access the specific file `archive/data/0` in the
Torch ZIP, it would start at `327378944`.

It worked on the `.pth` and `.pt` LLM models I have.

Source: Trust me bro.

## Usage

Look at the examples in [examples](examples/):
* `dump_raw.rs` — Dumps the raw Pickle opcodes from a file.
* `dump.rs` — Dumps the `Value`s from a file.
* `dump_torch.rs` — Dumps the tensor metadata for a PyTorch file.

Note that for the last one you'll need to have the `torch` feature enabled.
You can also run the example like:

    cargo run --features torch --example dump_raw -- /path/to/torchfile.pth

**Warning**: Don't try to run `dump` or `dump_raw` on a Torch file. It'll
try to load the entire massive file as a pickle.

## Caveats

This isn't very well tested. It probably won't delete all
your cat pictures when it fails, but it could fail.

Also, be careful of recursive Pickle files. It won't crash,
but it'll loop until it hits the recursion limit (about `250`)
and you'll get nested data up to that limit.

If you need to handle large Pickle files, ideally don't use this
at all, but if you must then you'd probably be better off
using the single Pickle op Nom parser in `src/parsers.rs`.
