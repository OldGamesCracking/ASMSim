**Note: This is a purely vibe coded library, I take no responsability for it! Do whatever you like with it, but use it on you own risk only.**

# ASMSim - A x86 assembly code simplifier/deobfuscator

**Background**

While there exist many great tools out there to lift assembly code to some IR or decompile it, I was unable to find a good library that was capable of doing a seemingly simple task: Simplify x86 code filled with junk.

Since it's the dawn of Artificial Intelligence, why not throw a few million tokens to one of them and let the computers do the heavy lifting (and unlifting, duh)?

## Capabilities

* Lifts x86-32 with iced-x86 into a hash-consed symbolic term DAG (constant folding,
  algebraic rewrites, exact flag semantics) and follows the stub's **real** control
  flow, so bytes skipped by a `jmp +1` never become instructions.
* **Conditional branches**: both arms are analysed and merged where they rejoin. The
  branch stays in the output with its two sides simplified (whatever computes its
  flags is kept); a condition that folds to a constant loses its dead arm. A stub may
  have several exits.
* **Loops**: a backward branch is emitted verbatim and the analysis continues behind it
  with the loop's results unknown, so a spin lock no longer needs `--keep` to be
  simplified.
* **Slicing**: backward liveness from the outputs you name plus whatever the terminating
  instruction reads and every observable memory write; runs of instructions that only
  compute a literal collapse into that literal. A store that writes back exactly the
  bytes it just read (`add [ecx], 0`, and the `push`/`xor`/`add`/`pop` filler around it)
  writes nothing observable and is dropped.
* **Verification**: original and simplified blob run side by side in Unicorn from
  identical randomised states (deterministic memory *and* a deterministic `rdtsc`),
  compared per exit, with a report of which exits were exercised. Memory is compared by
  content, so dropping a byte-identical store is not reported as a difference.
* **Output modes**: `inplace` (same addresses, dropped bytes become NOPs - patchable
  into the original image or a live process) and `compact` (survivors only, branches
  relaid out).
* **Sources**: a snapshot file, or a live x64dbg session through x64dbg-automate
  (read-only attach, or patch the paused target back in place).

## Usage

```python
from asmsim.deob import BytesSource, simplify
from asmsim.verify import verify

src = BytesSource(open("hunk.bin", "rb").read(), 0x1141000)
res = simplify(src, 0x114756A, regs_in=["ebx"], regs_out=["ebx"])
print(res.summary())
print(res.listing())
print(verify(src, res, runs=100))
```

The command line front end wraps the same call, plus patching:

```
python tools/deob.py --bin tests/fixtures/vmcode.bin --base 0x1141000 \
    --start 0x114756A --regs-in ebx --regs-out ebx --mode compact \
    --verify 100 --out handler.bin --listing handler.asm

python tools/deob.py --attach --start 0x1141AE7 --regs-in ebx --regs-out ebx --patch-live
```

`--args argN` declares stack arguments at `[esp+4*N]`, `--mem ADDR=VAL` known memory
(which resolves data-dependent branches so the trace can continue), `--keep LO-HI`
ranges emitted byte for byte, `--verify N` runs the differential test, `--out` /
`--patch-out` write the blob or a patched copy of the image. `python tools/deob.py
--help` lists everything.
