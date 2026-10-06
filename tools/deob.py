"""Command line front end for the asmsim stub simplifier.

Examples
--------
Offline, from a memory snapshot::

    python tools/deob.py --bin tests/fixtures/vmcode.bin --base 0x1141000 \
        --start 0x1141AE7 --regs-in ebx --regs-out ebx --out handler1.bin

Live, against a running x64dbg session (read-only attach), reading the stub at
the address where EIP currently is::

    python tools/deob.py --attach --start 0x1141AE7 --regs-in ebx --regs-out ebx \
        --out handler1.bin --verify 200

The same, but writing the simplified stub back into the *paused* target (the
dropped instructions become NOPs, so all internal addresses stay valid)::

    python tools/deob.py --attach --start 0x1141AE7 --regs-in ebx --regs-out ebx \
        --patch-live

Stubs that take their parameters on the stack declare them with ``--args``::

    python tools/deob.py --bin target.bin --base 0x400000 --start 0x401000 \
        --args arg0,arg1 --regs-out eax --verify 200

``argN`` is the dword at ``[entry_esp + 4*N]`` (so ``arg0`` is right at the entry
ESP - a stub entered through a ``call`` would name its first real argument ``arg1``,
and a ``pop``/``ret`` executed at the entry ESP reads the ``arg0`` slot).  Reads of
those slots show up as ``in:argN``; give a value (``arg0=0x1234``) to have it folded
into the output - such a blob then only works for callers passing that value.

``--mem`` declares what memory holds at an absolute address (a dword each), which is
what gets the trace past a data dependent branch such as a spin lock - and it is
seeded into both emulator runs, so the verification stays honest.

A declared value resolves the branch *statically*, which also means the never taken
side is dropped.  Conditional branches and loops are followed automatically now:
both arms of a branch are simplified and merged where they meet again, and a loop is
emitted verbatim (like ``--keep``) with the analysis continuing behind it with
unknown results - so a spin lock needs ``--mem`` only to make the *verifier* get
past the wait, not to analyse it::

    python tools/deob.py --bin target.bin --base 0x1141000 --start 0x1141980 \
        --args arg0,arg1 --mem 0x14a0240=1 --regs-out eax,edx --verify 200

``--keep`` is still the way to pin a range byte for byte, e.g. to run a known
``--mem`` value into a waiting loop::

    python tools/deob.py --bin target.bin --base 0x1141000 --start 0x1141980 \
        --args arg0,arg1 --mem 0x14a0240=1 --keep 0x1141987-0x114199d \
        --regs-out eax,edx --verify 200

``--unsafe-stores`` selects the smaller dead-store elimination that may drop a
store a surviving load still reads; the safe variant is the default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asmsim.deob import BadInstruction, BytesSource, RemoteSource, StubError, is_terminator, simplify


def parse_regs(text: str) -> dict:
    """``ebx`` / ``ebx,ecx`` / ``ebx=0x1234`` -> {name: value or None}"""
    out: dict = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        if "=" in part:
            name, val = part.split("=", 1)
            out[name.strip().lower()] = int(val, 0)
        else:
            out[part.lower()] = None
    return out


def make_source(args):
    if args.bin:
        hunk = Path(args.bin).read_bytes()
        return BytesSource(hunk, int(args.base, 0))
    import psutil
    from x64dbg_automate import X64DbgClient

    sessions = X64DbgClient.list_sessions()
    if not sessions:
        raise SystemExit("no x64dbg-automate session found - start x64dbg (with the plugin) first")
    if args.attach is not None and args.attach != 0:
        sess = next((s for s in sessions if s.pid == args.attach), None)
        if sess is None:
            raise SystemExit(f"no session with pid {args.attach} (have {[s.pid for s in sessions]})")
    else:
        sess = sessions[0]
    client = X64DbgClient(psutil.Process(sess.pid).exe())
    client.attach_session(sess.pid)
    print(f"# attached to pid {sess.pid}", file=sys.stderr)
    return RemoteSource(client)


def patch_live(source, res) -> str | None:
    """Write the simplified stub over the original in the attached process.

    Returns an error message, or None when the bytes are in place and read back
    identical.  ``--patch-live`` runs after ``--verify``, so the differential test
    still compares against the untouched original.
    """
    if not isinstance(source, RemoteSource):
        return "--patch-live needs --attach (there is no live process to patch)"
    if source.is_running():
        return "the target is running - pause it first, patching executing code can crash it"
    data = res.patch_bytes()
    if not source.write(res.start, data):
        return f"write_memory failed at {res.start:#x}"
    if source.read(res.start, len(data)) != data:
        return f"the bytes read back at {res.start:#x} do not match what was written"
    term = res.terminator_ip_out
    where = f", terminates at {term:#x}" if term is not None else ""
    print(f"# patched {len(data)} bytes at {res.start:#x}..{res.start + len(data):#x} "
          f"(mode={res.mode}{where})")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="simplify / de-obfuscate a straight-line x86-32 stub")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bin", help="snapshot file to read the stub from")
    src.add_argument("--attach", nargs="?", type=int, const=0,
                     help="attach to a live x64dbg-automate session (optional pid)")
    ap.add_argument("--base", default="0", help="load address of --bin (default 0)")
    ap.add_argument("--start", required=True, help="stub entry address")
    ap.add_argument("--regs-in", default="", help="input registers, e.g. ebx or ebx=0x1234,ecx")
    ap.add_argument("--regs-out", default="", help="output registers to preserve, e.g. ebx")
    ap.add_argument("--args", default="",
                    help="incoming stack arguments arg0, arg1, ... at [esp+0], [esp+4], ... "
                         "(name=0x1234 for a known value; a concrete value is folded into "
                         "the output, so --patch-out/--patch-live then only fit callers "
                         "that pass it)")
    ap.add_argument("--mem", default="",
                    help="known memory as address=value pairs, e.g. 0x14a0240=1 or 0x14a0244 "
                         "(a dword each; a known value resolves data dependent branches so "
                         "the trace can continue past them)")
    ap.add_argument("--keep", default="",
                    help="byte range(s) LO-HI emitted verbatim, e.g. 0x1141987-0x114199d for a "
                         "spin lock: the trace still steps through them (so --mem/--args keep "
                         "working) but nothing inside is rewritten or dropped")
    ap.add_argument("--mode", choices=("inplace", "compact"), default="inplace")
    ap.add_argument("--out", help="write the simplified binary here")
    ap.add_argument("--patch-out", metavar="FILE",
                    help="write a copy of --bin with the simplified stub patched in at --start")
    ap.add_argument("--listing", help="write the listing here (default: stdout)")
    ap.add_argument("--hide-dropped", action="store_true", help="listing: do not show removed instructions")
    ap.add_argument("--annotate", type=int, default=2,
                    help="listing: expression depth for written registers (0 = off)")
    ap.add_argument("--expr-depth", type=int, default=8, help="depth of the output expressions")
    ap.add_argument("--max-instructions", type=int, default=20000)
    ap.add_argument("--drop-unread-stores", action="store_true",
                    help="also drop memory stores no later load reads (changes memory semantics)")
    ap.add_argument("--unsafe-stores", action="store_true",
                    help="also drop stores a surviving load still reads at runtime "
                         "(smaller output, but unsound dead-store elimination)")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="run N differential Unicorn executions of original vs simplified")
    ap.add_argument("--patch-live", action="store_true",
                    help="write the simplified stub back into the attached process "
                         "(the target must be paused)")
    args = ap.parse_args()

    start = int(args.start, 0)
    source = make_source(args)
    rep = None
    try:
        res = simplify(source, start,
                       regs_in=parse_regs(args.regs_in), regs_out=parse_regs(args.regs_out),
                       is_end=is_terminator, mode=args.mode,
                       max_instructions=args.max_instructions,
                       keep_unread_stores=not args.drop_unread_stores,
                       protect_loaded_stores=not args.unsafe_stores,
                       args=parse_regs(args.args), mem=parse_regs(args.mem),
                       keep=args.keep or None)
        if args.verify:
            from asmsim.verify import verify
            rep = verify(source, res, runs=args.verify,
                         compare_memory=not args.drop_unread_stores)
        if args.patch_live:
            err = patch_live(source, res)
            if err:
                print(f"error: {err}", file=sys.stderr)
                return 2
    except (StubError, ValueError) as e:      # ValueError: a bad register/argument name
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        source.close()

    if (res.concrete_args or res.concrete_mem) and (args.patch_live or args.patch_out):
        print("# note: declared values (concrete args/memory) are folded into the output - the "
              "patched code only behaves correctly where they hold", file=sys.stderr)
    print(res.summary())
    print()
    listing = res.listing(show_dropped=not args.hide_dropped, annotate=args.annotate)
    if args.listing:
        Path(args.listing).write_text(listing + "\n", encoding="utf-8")
        print(f"# listing written to {args.listing}")
    else:
        print(listing)
    if args.out:
        Path(args.out).write_bytes(res.code)
        print(f"# {len(res.code)} bytes written to {args.out}")
    if args.patch_out:
        if not args.bin:
            print("warning: --patch-out needs --bin (nothing to patch into)", file=sys.stderr)
        else:
            image = bytearray(Path(args.bin).read_bytes())
            off = start - int(args.base, 0)
            if off < 0 or off + len(res.code) > len(image):
                print("error: the stub does not fit into the image", file=sys.stderr)
                return 2
            image[off:off + len(res.code)] = res.code
            Path(args.patch_out).write_bytes(bytes(image))
            print(f"# patched image written to {args.patch_out} "
                  f"({len(res.code)} bytes at {start:#x})")

    if rep is not None:
        print()
        print(rep)
        return 0 if rep.ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
