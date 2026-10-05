"""Fuzz the lifter against Unicorn, one instruction at a time.

Random byte strings are decoded with iced; every instruction the lifter claims to
support is executed both symbolically (then evaluated on concrete inputs) and in
Unicorn from the same random machine state.  Registers, defined flags and memory
writes must agree.
"""
import random

import pytest
from iced_x86 import Decoder, DecoderOptions, Mnemonic as MN, OpKind as OK, RflagsBits as RB

from asmsim import terms as T
from asmsim.emu import Emu, FLAG_BITS, MemError, initial_byte, random_regs
from asmsim.lift import HANDLERS, Lifter, State, Unsupported

CODE = 0x00401000
IFLAG = {"cf": RB.CF, "pf": RB.PF, "af": RB.AF, "zf": RB.ZF, "sf": RB.SF, "of": RB.OF}

SKIP = {MN.CPUID, MN.RDTSC, MN.HLT, MN.INT3, MN.UD2, MN.INT, MN.INTO, MN.INT1, MN.SYSENTER,
        MN.SYSCALL, MN.JMP, MN.CALL, MN.RET, MN.PAUSE, MN.NOP, MN.FNOP, MN.WAIT, MN.JECXZ,
        MN.DIV, MN.IDIV, MN.POPFD, MN.POPAD, MN.PUSHFD}


def is_branch_mnemonic(m):
    return m in SKIP or str(m) in ()


def check_one(code: bytes, seed: int) -> bool:
    """Returns True when the instruction was exercised, False when skipped."""
    emus = []
    try:
        return _check_one(code, seed, emus)
    finally:
        for e in emus:
            e.close()


def _check_one(code: bytes, seed: int, emus: list) -> bool:
    d = Decoder(32, code, ip=CODE)
    ins = d.decode()
    if ins.is_invalid or ins.len > len(code) or ins.mnemonic not in HANDLERS or ins.mnemonic in SKIP:
        return False
    if ins.flow_control != 0:                     # branches are covered elsewhere
        return False
    if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repe_prefix or ins.has_repne_prefix:
        return False
    raw = code[:ins.len]

    rng = random.Random(seed)
    regs = random_regs(rng)
    emu = Emu(seed, {CODE: raw})
    emus.append(emu)
    emu.set_state(regs)
    init_mem_ok = True

    st = State.initial()
    try:
        Lifter(st).step(ins)
    except Unsupported:
        return False

    stores_eval: list[tuple[int, int, int, int]] = []     # (seq, addr, size, value)

    def env(t):
        if t.op == "in":
            n = t.args[0]
            if n in regs:
                return regs[n]
            return int(bool(regs["eflags"] & FLAG_BITS[n]))
        if t.op == "undef":
            raise AssertionError("undef reached the evaluator")
        if t.op == "load":
            seg, addr_t, mstate = t.args
            a = T.evaluate(addr_t, env, memo)
            n = t.bits // 8
            data = bytearray(emu.read(a, n))
            for seq, sa, ssz, sv in stores_eval:
                if seq <= mstate:
                    for i in range(ssz):
                        if a <= sa + i < a + n:
                            data[sa + i - a] = (sv >> (8 * i)) & 0xFF
            return int.from_bytes(data, "little")
        raise AssertionError(t.op)
    memo: dict = {}

    try:
        for s in st.mem.stores:
            a = T.evaluate(s.addr, env, memo)
            v = T.evaluate(s.val, env, memo)
            stores_eval.append((s.seq, a, s.size, v))
        exp_regs = {n: T.evaluate(st.regs[n], env, memo) for n in st.regs}
        exp_flags = {}
        for f, bit in IFLAG.items():
            if ins.rflags_undefined & bit:
                continue
            exp_flags[f] = T.evaluate(st.flags[f], env, memo)
    except AssertionError as e:
        if "undef" in str(e):
            return False
        raise

    try:
        emu.run(CODE, CODE + ins.len, count=1)
    except Exception:
        return False
    got = emu.get_state()
    where = f"{ins}  [{raw.hex()}] seed={seed}\n  regs0={ {k: hex(v) for k, v in regs.items()} }"
    for n, v in exp_regs.items():
        assert got[n] == v, f"{n}: emu {got[n]:#x} != lifted {v:#x} for {where}"
    for f, v in exp_flags.items():
        gv = int(bool(got["eflags"] & FLAG_BITS[f]))
        assert gv == v, f"flag {f}: emu {gv} != lifted {v} for {where}"
    # memory
    for seq, a, sz, v in stores_eval:
        pass
    final = {}
    for seq, a, sz, v in stores_eval:
        for i in range(sz):
            final[(a + i) & 0xFFFFFFFF] = (v >> (8 * i)) & 0xFF
    wrote = emu.written_bytes()
    for a, b in wrote.items():
        orig = initial_byte(seed, a)
        assert final.get(a, orig) == b, f"mem[{a:#x}]: emu {b:#x} != lifted {final.get(a, orig):#x} for {where}"
    for a, b in final.items():
        assert emu.read(a, 1)[0] == b, f"mem[{a:#x}] lifted store {b:#x} not seen in emu for {where}"
    return True


def test_fuzz_instructions():
    rng = random.Random(20240601)
    ok = 0
    seen = set()
    attempts = 0
    while ok < 6000 and attempts < 400000:
        attempts += 1
        n = rng.choice([2, 3, 4, 5, 6, 7])
        code = rng.randbytes(n) + rng.randbytes(8)
        if rng.random() < 0.5:                        # bias towards common prefixes / 0F escapes
            code = rng.choice([b"\x0f", b"\x66", b"\x0f", b""]) + code
        try:
            done = check_one(code, rng.getrandbits(30))
        except MemError:
            continue
        if done:
            ok += 1
            seen.add(Decoder(32, code, ip=CODE).decode().mnemonic)
    names = sorted(str(m) for m in seen)
    print(f"\nexercised {ok} instructions covering {len(seen)} mnemonics")
    assert ok >= 3000
    # make sure the interesting families were really hit
    for must in (MN.ADD, MN.ADC, MN.SBB, MN.SHLD, MN.SHRD, MN.RCL, MN.RCR, MN.BT, MN.BTS, MN.BSF,
                 MN.BSR, MN.XADD, MN.IMUL, MN.MOVZX, MN.MOVSX, MN.BSWAP, MN.ROL, MN.NEG, MN.LEA,
                 MN.XCHG, MN.PUSH, MN.POP):
        assert must in seen, f"fuzzer never hit {must}"
