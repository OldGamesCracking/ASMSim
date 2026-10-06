"""Differential verification of a simplified stub against the original.

Both versions are executed in the Unicorn reference emulator (the same one the
lifter is fuzzed against) from identical random machine states.  Memory is the
real snapshot for the code image; everything else is lazily mapped deterministic
garbage, so both runs see exactly the same bytes - including pointers that point
into nowhere.

What is compared:

* every register in ``regs_out`` plus everything the terminating instruction
  reads (``jmp eax`` -> eax, ``ret`` -> esp),
* every memory byte written outside the stub's own new stack frame (memory below
  the entry ESP is the stub's private frame; that is the one assumption the
  simplifier's dead-store analysis makes).

A stub with several exits is verified per exit: each run stops at whichever exit
it reaches and the two programs must agree on *which* exit that was, on the
registers that exit's terminator reads, and on the memory.  The report says how
many runs took each exit, so a branch that one side never walks cannot look like a
clean pass.

Registers that are neither outputs nor exits are reported separately when they
differ: dropping them is the point of the simplification, not a failure.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Sequence

from .deob import Result
from .emu import REGS, Emu, random_regs

PAGE = 0x1000


@dataclasses.dataclass
class Report:
    ok: bool
    runs: int
    skipped: int
    mismatches: list[str]
    discarded: list[str]
    compare_memory: bool = True
    coverage: dict = dataclasses.field(default_factory=dict)   # exit index -> runs that took it

    def __str__(self) -> str:
        lines = [f"verify: {self.runs} randomised runs, {self.skipped} skipped"]
        if self.mismatches:
            lines.append(f"  MISMATCH in {len(self.mismatches)} run(s):")
            lines += [f"    {m}" for m in self.mismatches[:12]]
            if len(self.mismatches) > 12:
                lines.append(f"    ... and {len(self.mismatches) - 12} more")
        else:
            lines.append("  outputs" + (" and observable memory" if self.compare_memory else "")
                         + " match the original")
        if len(self.coverage) > 1:
            lines.append("  exits taken: "
                         + ", ".join(f"#{k}: {v}" for k, v in sorted(self.coverage.items())))
            never = [k for k, v in sorted(self.coverage.items()) if v == 0]
            if never:
                lines.append(f"  !! exit(s) {', '.join('#' + str(k) for k in never)} were never "
                             f"taken - the simplified code may hide a difference there")
        if self.discarded:
            lines.append(f"  discarded registers (by design, not outputs): {', '.join(self.discarded)}")
        return "\n".join(lines)


def _image(src, lo: int, hi: int) -> dict[int, bytes]:
    """Real bytes for every page of [lo, hi) the source can provide."""
    out: dict[int, bytes] = {}
    for page in range(lo & ~(PAGE - 1), (hi + PAGE - 1) & ~(PAGE - 1), PAGE):
        try:
            out[page] = src.read(page, PAGE)
        except Exception:
            pass
    return out


def _prepare(image: dict[int, bytes], stops: dict[int, int], seed: int,
             max_insns: int = 200000, args: Sequence[tuple[int, int]] = (),
             mem: Sequence[tuple[int, int]] = ()) -> Emu:
    """An emulator with the code image mapped that stops right *before* any exit.

    ``stops`` maps an exit's address to its key; ``execute`` runs until one of them
    is reached and returns that key.  The stop is done by stepping one instruction
    at a time - unicorn's ``emu_stop()`` from a code hook only takes effect at a
    translation block boundary, which would execute the terminator (and everything
    after it in the same block) in one program but not the other.

    ``args`` are the concrete stack arguments as ``(offset from entry ESP, value)``
    and ``mem`` the known memory contents as ``(absolute address, value)``: both are
    written into memory for every run, so the emulator sees the same values the
    symbolic side assumed."""
    emu = Emu(seed)
    for base, data in image.items():
        emu.map_code(base, data)
    eip = REGS["eip"]

    def run(regs: dict, start: int, code: bytes | None = None, run_seed: int = seed) -> int:
        emu.reset(run_seed)
        if code is not None:
            emu.write_image(start, code)
        emu.set_state(regs)
        for off, value in args:                     # seeding is not a program write
            addr = (regs["esp"] + off) & 0xFFFFFFFF
            emu.ensure(addr, 4)
            emu.write_image(addr, value.to_bytes(4, "little"))
        for addr, value in mem:
            emu.ensure(addr, 4)
            emu.write_image(addr, (value & 0xFFFFFFFF).to_bytes(4, "little"))
        emu.write_log.clear()
        ip = start
        for _ in range(max_insns):
            key = stops.get(ip)
            if key is not None:
                return key
            emu.uc.emu_start(ip, 0, count=1)
            ip = emu.uc.reg_read(eip)
        raise RuntimeError(f"never reached any of {[hex(x) for x in stops]}")

    emu.execute = run                                   # type: ignore[attr-defined]
    return emu


def verify(src, res: Result, runs: int = 100, seed0: int = 0x5EED,
           max_insns: int = 200000, compare_memory: bool = True) -> Report:
    if not res.exits:
        return Report(False, 0, 0, ["analysis stopped early - nothing to verify"], [])
    stop_orig = {e.step.ip: i for i, e in enumerate(res.exits)}
    stop_simp = {e.out_ip: i for i, e in enumerate(res.exits)}
    if (None in stop_simp or len(stop_orig) != len(res.exits)
            or len(stop_simp) != len(res.exits)):
        return Report(False, 0, 0, ["cannot locate the exits in the output"], [])
    lo = min(min(e.step.ip for e in res.exits), res.start)
    hi = max(max(e.step.end for e in res.exits), res.end)
    image = _image(src, lo - 0x2000, hi + 0x2000)

    others = ("eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp")
    mismatches: list[str] = []
    discarded: set[str] = set()
    coverage: dict = {i: 0 for i in range(len(res.exits))}
    skipped = 0

    concrete_args = sorted(res.concrete_args.items())
    concrete_mem = sorted(res.concrete_mem.items())
    a = _prepare(image, stop_orig, seed0, max_insns, args=concrete_args, mem=concrete_mem)
    b = _prepare(image, stop_simp, seed0, max_insns, args=concrete_args, mem=concrete_mem)
    try:
        for i in range(runs):
            seed = seed0 + i
            regs = random_regs(random.Random(seed))
            regs.update(res.concrete)          # the run assumes these entry values
            try:
                which_a = a.execute(regs, res.start)
                which_b = b.execute(regs, res.start, code=res.code)
            except Exception:
                skipped += 1
                continue
            sa, sb = a.get_state(), b.get_state()
            where = f"seed={seed:#x} regs0={{{', '.join(f'{k}={v:#x}' for k, v in sorted(regs.items()))}}}"
            if which_a != which_b:
                mismatches.append(f"exit taken: original #{which_a} != simplified "
                                  f"#{which_b} ({where})")
            else:
                coverage[which_a] += 1
                compare = res.exits[which_a].compare
                for name in compare:
                    if sa[name] != sb[name]:
                        mismatches.append(f"{name}: original {sa[name]:#x} != "
                                          f"simplified {sb[name]:#x} ({where})")
            for name in others:
                if sa[name] != sb[name]:
                    discarded.add(name)
            if sa["eflags"] != sb["eflags"]:
                discarded.add("eflags")
            if compare_memory:
                esp0 = regs["esp"]
                wa = {ad: v for ad, v in a.written_bytes().items() if ad >= esp0}
                wb = {ad: v for ad, v in b.written_bytes().items() if ad >= esp0}
                for addr in sorted(set(wa) | set(wb)):
                    if wa.get(addr) != wb.get(addr):
                        mismatches.append(f"mem[{addr:#x}]: original {wa.get(addr)} != "
                                          f"simplified {wb.get(addr)} ({where})")
    finally:
        a.close()
        b.close()

    if runs and skipped >= runs:
        # every execution failed before the comparison - that is not a pass
        mismatches.append(f"all {runs} runs were skipped (nothing to compare)")
    return Report(ok=not mismatches, runs=runs - skipped, skipped=skipped,
                  mismatches=mismatches, discarded=sorted(discarded),
                  compare_memory=compare_memory, coverage=coverage)
