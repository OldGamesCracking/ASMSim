"""x86-32 instruction semantics on the symbolic state.

``Lifter.step(ins)`` applies one decoded instruction to a ``State`` (registers,
flags, memory) and reports the control-flow outcome as a ``Flow``.  Everything is
expressed with the term DAG from :mod:`asmsim.terms`; flag bits are plain 1-bit
terms (so dead flags simply disappear) plus an *origin* tag that lets conditional
jumps be recovered as readable comparisons (``cmp a,b ; jl`` -> ``slt(a, b)``).
"""
from __future__ import annotations

from iced_x86 import Code, ConditionCode as CC
from iced_x86 import Mnemonic as MN
from iced_x86 import OpKind as OK
from iced_x86 import MemorySizeExt, Register as R

from . import terms as T
from .memory import Memory
from .terms import (add, sub, mul, and_, or_, xor, not_, neg, shl, shr, sar, rol, ror, zext, sext,
                    trunc, ins as ins_, ite, eq, ult, slt, par, const, undef, TRUE, FALSE)

REG32 = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"]
FLAGS = ["cf", "pf", "af", "zf", "sf", "of", "df"]


class Unsupported(Exception):
    """Raised for instructions the symbolic engine does not model."""

    def __init__(self, ins, why: str = ""):
        self.ins, self.why = ins, why
        super().__init__(f"unsupported instruction at {ins.ip:#x}: {ins} {why}".strip())


# --- register table: iced Register -> (full 32 bit name, bits, bit shift) ----------------
REGINFO: dict[int, tuple[str, int, int]] = {}
for _i, _n in enumerate(REG32):
    REGINFO[getattr(R, _n.upper())] = (_n, 32, 0)
    REGINFO[getattr(R, _n[1:].upper())] = (_n, 16, 0)
for _i, _n in enumerate(["al", "cl", "dl", "bl"]):
    REGINFO[getattr(R, _n.upper())] = (REG32[_i], 8, 0)
for _i, _n in enumerate(["ah", "ch", "dh", "bh"]):
    REGINFO[getattr(R, _n.upper())] = (REG32[_i], 8, 8)

# segment registers: never compared or folded on, only pushed and popped.  Their
# selector values are not part of the model, so a ``push seg`` pushes an unknown.
SEGINFO = {getattr(R, n.upper()): n for n in ("ds", "es", "ss", "cs", "fs", "gs")}

CCNAMES = {getattr(CC, n.upper()): n for n in
           ("o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g")}
CC_FLAGS = {"o": ("of",), "no": ("of",), "b": ("cf",), "ae": ("cf",), "e": ("zf",), "ne": ("zf",),
            "be": ("cf", "zf"), "a": ("cf", "zf"), "s": ("sf",), "ns": ("sf",), "p": ("pf",),
            "np": ("pf",), "l": ("sf", "of"), "ge": ("sf", "of"), "le": ("zf", "sf", "of"),
            "g": ("zf", "sf", "of")}


# ===========================================================================================
# state
# ===========================================================================================
class State:
    def __init__(self, isolate_new_stack: bool = True):
        self.regs: dict[str, T.Term] = {}
        self.flags: dict[str, T.Term] = {}
        self.forigin: dict[str, tuple | None] = {}
        self.mem = Memory(isolate_new_stack)

    @classmethod
    def initial(cls, concrete: dict[str, int] | None = None, isolate_new_stack: bool = True,
                args: dict[int, T.Term] | None = None,
                mem: dict[int, T.Term] | None = None) -> "State":
        s = cls(isolate_new_stack)
        for n in REG32:
            t = T.inp(n)
            t.hint = n
            s.regs[n] = t
        for n, v in (concrete or {}).items():
            s.regs[n] = const(v)
        s.mem.arg_slots.update(args or {})         # entry ESP-relative stack arguments
        s.mem.mem_slots.update(mem or {})          # known contents at absolute addresses
        for f in FLAGS:
            s.flags[f] = T.inp(f, 1)
            s.forigin[f] = None
        s.flags["df"] = FALSE                      # the Windows ABI guarantees DF == 0
        return s

    def copy(self) -> "State":
        s = State(self.mem.isolate_new_stack)
        s.regs = dict(self.regs)
        s.flags = dict(self.flags)
        s.forigin = dict(self.forigin)
        s.mem = self.mem.copy()
        return s

    # -- registers -----------------------------------------------------------------------
    def read_reg(self, reg: int) -> T.Term:
        name, bits, shift = REGINFO[reg]
        full = self.regs[name]
        if bits == 32:
            return full
        if shift:
            return trunc(shr(full, const(shift, 8)), bits)
        return trunc(full, bits)

    def write_reg(self, reg: int, val: T.Term) -> None:
        name, bits, shift = REGINFO[reg]
        assert val.bits == bits, (reg, val)
        new = val if bits == 32 else ins_(self.regs[name], val, shift)
        if new.hint is None:
            new.hint = name
        self.regs[name] = new

    @property
    def esp(self) -> T.Term:
        return self.regs["esp"]


# ===========================================================================================
# control flow outcome
# ===========================================================================================
class Flow:
    __slots__ = ("kind", "target", "cond", "taken", "fall", "term")

    def __init__(self, kind, target=None, cond=None, taken=None, fall=None, term=None):
        self.kind, self.target, self.cond, self.taken, self.fall, self.term = kind, target, cond, taken, fall, term


NEXT = Flow("next")


def jump(addr: int) -> Flow:
    return Flow("jump", target=addr)


# ===========================================================================================
# operands
# ===========================================================================================
class RegOp:
    def __init__(self, st: State, reg: int):
        self.st, self.reg = st, reg
        self.bits = REGINFO[reg][1]

    def read(self):
        return self.st.read_reg(self.reg)

    def write(self, v):
        self.st.write_reg(self.reg, v)


class MemOp:
    def __init__(self, st: State, seg, addr: T.Term, nbytes: int):
        self.st, self.seg, self.addr, self.nbytes = st, seg, addr, nbytes
        self.bits = nbytes * 8

    def read(self):
        return self.st.mem.load(self.seg, self.addr, self.nbytes)

    def write(self, v):
        self.st.mem.store(self.seg, self.addr, self.nbytes, v)


class ImmOp:
    def __init__(self, value: int, bits: int):
        self.v, self.bits = value & T.MASK[bits], bits

    def read(self):
        return const(self.v, self.bits)

    def write(self, v):  # pragma: no cover
        raise AssertionError("write to immediate")


_IMM_KINDS = {OK.IMMEDIATE8, OK.IMMEDIATE16, OK.IMMEDIATE32, OK.IMMEDIATE8TO16, OK.IMMEDIATE8TO32,
              OK.IMMEDIATE8_2ND}


# ===========================================================================================
# the lifter
# ===========================================================================================
HANDLERS: dict[int, callable] = {}


def handles(*names):
    def deco(fn):
        for n in names:
            HANDLERS[getattr(MN, n)] = fn
        return fn
    return deco


def msb(x: T.Term) -> T.Term:
    return slt(x, const(0, x.bits))


def lsb(x: T.Term) -> T.Term:
    return trunc(x, 1)


def bit_at(x: T.Term, k) -> T.Term:
    """Bit ``k`` (int or 8-bit term) of ``x``."""
    kt = const(k, 8) if isinstance(k, int) else k
    return trunc(shr(x, kt), 1)


def lsh(x: T.Term, k: int) -> T.Term:
    return const(0, x.bits) if k >= x.bits else shl(x, const(k, 8))


def rsh(x: T.Term, k: int) -> T.Term:
    return const(0, x.bits) if k >= x.bits else shr(x, const(k, 8))


class Lifter:
    def __init__(self, st: State):
        self.st = st

    # -- operand plumbing -----------------------------------------------------------------
    def address(self, ins) -> T.Term:
        if ins.memory_base != R.NONE and REGINFO.get(ins.memory_base, (None, 0))[1] != 32:
            raise Unsupported(ins, "16 bit addressing")
        acc = None
        if ins.memory_base != R.NONE:
            acc = self.st.read_reg(ins.memory_base)
        if ins.memory_index != R.NONE:
            idx = self.st.read_reg(ins.memory_index)
            sc = ins.memory_index_scale
            if sc > 1:
                idx = shl(idx, const(sc.bit_length() - 1, 8))
            acc = idx if acc is None else add(acc, idx)
        disp = const(ins.memory_displacement & 0xFFFFFFFF)
        return disp if acc is None else add(acc, disp)

    def operand(self, ins, k: int, size_hint: int | None = None):
        kind = ins.op_kind(k)
        if kind == OK.REGISTER:
            reg = ins.op_register(k)
            if reg not in REGINFO:
                raise Unsupported(ins, f"register {reg}")
            return RegOp(self.st, reg)
        if kind == OK.MEMORY:
            nbytes = MemorySizeExt.size(ins.memory_size)
            if nbytes not in (1, 2, 4):
                raise Unsupported(ins, f"memory operand of {nbytes} bytes")
            seg = {R.FS: "fs", R.GS: "gs"}.get(ins.segment_prefix)
            return MemOp(self.st, seg, self.address(ins), nbytes)
        if kind in _IMM_KINDS:
            return ImmOp(ins.immediate(k), size_hint or 32)
        raise Unsupported(ins, f"operand kind {kind}")

    def cond_for(self, ins) -> T.Term:
        return self.cond(CCNAMES[ins.condition_code])

    # -- flags -----------------------------------------------------------------------------
    def setf(self, origin=None, **fl):
        for name, val in fl.items():
            self.st.flags[name] = val
            self.st.forigin[name] = origin

    def zsp(self, res: T.Term):
        return dict(zf=eq(res, const(0, res.bits)), sf=msb(res), pf=par(res))

    def flags_logic(self, res):
        self.setf(("logic", res), cf=FALSE, of=FALSE, af=FALSE, **self.zsp(res))

    def flags_addsub(self, kind, a, b, res, cin=None):
        if kind == "add":
            cf = ult(res, a) if cin is None else ite(cin, not_(ult(a, res)), ult(res, a))
            of = msb(and_(xor(a, res), xor(b, res)))
        else:
            cf = ult(a, b) if cin is None else ite(cin, not_(ult(b, a)), ult(a, b))
            of = msb(and_(xor(a, b), xor(a, res)))
        af = trunc(shr(xor(xor(a, b), res), const(4, 8)), 1)
        origin = (kind, a, b, res) if cin is None else ("res", res)
        self.setf(origin, cf=cf, of=of, af=af, **self.zsp(res))

    def _zero_count(self, cnt: T.Term) -> T.Term:
        return eq(and_(cnt, const(31, 8)), const(0, 8))

    def _keep_if_zero(self, zero, **fl):
        """flags that stay unchanged when the (variable) shift count is 0"""
        old = dict(self.st.flags)
        self.setf(None, **{n: ite(zero, old[n], v) for n, v in fl.items()})

    # -- condition codes -----------------------------------------------------------------------
    def cond(self, cc: str) -> T.Term:
        st = self.st
        names = CC_FLAGS[cc]
        orgs = {id(st.forigin.get(n)) for n in names}
        org = st.forigin.get(names[0])
        if len(orgs) == 1 and org is not None:
            t = _cond_pattern(cc, org)
            if t is not None:
                return t
        f = st.flags
        negate = cc in ("no", "ns", "np", "ne", "ae", "a", "ge", "g")
        pos = {"o": "o", "no": "o", "b": "b", "ae": "b", "e": "e", "ne": "e", "be": "be", "a": "be",
               "s": "s", "ns": "s", "p": "p", "np": "p", "l": "l", "ge": "l", "le": "le", "g": "le"}[cc]
        t = {"o": lambda: f["of"], "b": lambda: f["cf"], "e": lambda: f["zf"],
             "be": lambda: or_(f["cf"], f["zf"]), "s": lambda: f["sf"], "p": lambda: f["pf"],
             "l": lambda: xor(f["sf"], f["of"]),
             "le": lambda: or_(f["zf"], xor(f["sf"], f["of"]))}[pos]()
        return not_(t) if negate else t

    # -- driver --------------------------------------------------------------------------------
    def step(self, ins) -> Flow:
        # the model has one thread and no memory ordering, so "lock" adds nothing -
        # the instruction itself is lifted as usual
        if ins.has_rep_prefix or ins.has_repe_prefix or ins.has_repne_prefix:
            if ins.mnemonic not in (MN.PAUSE,) and ins.mnemonic not in REP_MNEMONICS:
                raise Unsupported(ins, "rep prefix")
        fn = HANDLERS.get(ins.mnemonic)
        if fn is None:
            raise Unsupported(ins)
        r = fn(self, ins)
        return NEXT if r is None else r

    # -- stack ---------------------------------------------------------------------------------
    def push_val(self, val: T.Term):
        st = self.st
        esp = add(st.esp, const(-4))
        st.regs["esp"] = esp
        st.mem.store(None, esp, 4, val)

    def pop_val(self) -> T.Term:
        st = self.st
        v = st.mem.load(None, st.esp, 4)
        st.regs["esp"] = add(st.esp, const(4))
        return v


def _cond_pattern(cc: str, org):
    """Recover a readable predicate when all flags of ``cc`` come from one instruction."""
    kind = org[0]
    nt = not_

    def zero(x): return eq(x, const(0, x.bits))
    if kind == "sub":
        _, a, b, res = org
        m = {"e": lambda: eq(a, b), "ne": lambda: nt(eq(a, b)),
             "b": lambda: ult(a, b), "ae": lambda: nt(ult(a, b)),
             "be": lambda: nt(ult(b, a)), "a": lambda: ult(b, a),
             "l": lambda: slt(a, b), "ge": lambda: nt(slt(a, b)),
             "le": lambda: nt(slt(b, a)), "g": lambda: slt(b, a),
             "s": lambda: msb(res), "ns": lambda: nt(msb(res)),
             "p": lambda: par(res), "np": lambda: nt(par(res))}
    elif kind == "logic":
        _, res = org
        m = {"e": lambda: zero(res), "ne": lambda: nt(zero(res)),
             "s": lambda: msb(res), "ns": lambda: nt(msb(res)),
             "l": lambda: msb(res), "ge": lambda: nt(msb(res)),
             "le": lambda: or_(zero(res), msb(res)), "g": lambda: nt(or_(zero(res), msb(res))),
             "b": lambda: FALSE, "ae": lambda: TRUE, "be": lambda: zero(res),
             "a": lambda: nt(zero(res)), "o": lambda: FALSE, "no": lambda: TRUE,
             "p": lambda: par(res), "np": lambda: nt(par(res))}
    elif kind == "add":
        _, a, b, res = org
        m = {"e": lambda: zero(res), "ne": lambda: nt(zero(res)),
             "b": lambda: ult(res, a), "ae": lambda: nt(ult(res, a)),
             "be": lambda: or_(ult(res, a), zero(res)), "a": lambda: nt(or_(ult(res, a), zero(res))),
             "s": lambda: msb(res), "ns": lambda: nt(msb(res)),
             "p": lambda: par(res), "np": lambda: nt(par(res))}
    elif kind == "res":
        _, res = org
        m = {"e": lambda: zero(res), "ne": lambda: nt(zero(res)),
             "s": lambda: msb(res), "ns": lambda: nt(msb(res)),
             "p": lambda: par(res), "np": lambda: nt(par(res))}
    else:
        return None
    f = m.get(cc)
    return f() if f else None


# ===========================================================================================
# instruction handlers
# ===========================================================================================
@handles("NOP", "PAUSE", "FNOP", "WAIT")
def _nop(L, ins):
    return None


@handles("MOV")
def _mov(L, ins):
    if ins.op_kind(0) == OK.REGISTER and ins.op_register(0) not in REGINFO:
        raise Unsupported(ins, "segment/control register move")
    d = L.operand(ins, 0)
    s = L.operand(ins, 1, d.bits)
    d.write(s.read())


@handles("MOVZX")
def _movzx(L, ins):
    d = L.operand(ins, 0)
    s = L.operand(ins, 1)
    d.write(zext(s.read(), d.bits))


@handles("MOVSX")
def _movsx(L, ins):
    d = L.operand(ins, 0)
    s = L.operand(ins, 1)
    d.write(sext(s.read(), d.bits))


@handles("LEA")
def _lea(L, ins):
    d = L.operand(ins, 0)
    a = L.address(ins)
    d.write(a if d.bits == 32 else trunc(a, d.bits))


@handles("XCHG")
def _xchg(L, ins):
    d = L.operand(ins, 0)
    s = L.operand(ins, 1)
    a, b = d.read(), s.read()
    d.write(b)
    s.write(a)


@handles("XADD")
def _xadd(L, ins):
    d = L.operand(ins, 0)
    s = L.operand(ins, 1)
    a, b = d.read(), s.read()
    res = add(a, b)
    s.write(a)
    d.write(res)
    L.flags_addsub("add", a, b, res)


@handles("BSWAP")
def _bswap(L, ins):
    d = L.operand(ins, 0)
    if d.bits != 32:
        raise Unsupported(ins)
    d.write(T.bswap(d.read()))


def _alu(kind):
    def fn(L, ins):
        d = L.operand(ins, 0)
        s = L.operand(ins, 1, d.bits)
        a, b = d.read(), s.read()
        if kind in ("adc", "sbb"):
            cin = L.st.flags["cf"]
            if kind == "adc":
                res = add(add(a, b), zext(cin, a.bits))
            else:
                res = sub(sub(a, b), zext(cin, a.bits))
            L.flags_addsub("add" if kind == "adc" else "sub", a, b, res, cin)
            d.write(res)
        elif kind in ("add", "sub", "cmp"):
            res = add(a, b) if kind == "add" else sub(a, b)
            L.flags_addsub("add" if kind == "add" else "sub", a, b, res)
            if kind != "cmp":
                d.write(res)
        else:
            res = {"and": and_, "or": or_, "xor": xor, "test": and_}[kind](a, b)
            L.flags_logic(res)
            if kind != "test":
                d.write(res)
    return fn


for _n in ("add", "adc", "sub", "sbb", "and", "or", "xor", "cmp", "test"):
    HANDLERS[getattr(MN, _n.upper())] = _alu(_n)


@handles("INC", "DEC")
def _incdec(L, ins):
    d = L.operand(ins, 0)
    a = d.read()
    n = a.bits
    if ins.mnemonic == MN.INC:
        res = add(a, const(1, n))
        of = eq(res, const(1 << (n - 1), n))
        org = ("add", a, const(1, n), res)
        b = const(1, n)
    else:
        res = sub(a, const(1, n))
        of = eq(res, const((1 << (n - 1)) - 1, n))
        org = ("sub", a, const(1, n), res)
        b = const(1, n)
    af = trunc(shr(xor(xor(a, b), res), const(4, 8)), 1)
    L.setf(org, of=of, af=af, **L.zsp(res))      # CF is preserved
    d.write(res)


@handles("NEG")
def _neg(L, ins):
    d = L.operand(ins, 0)
    a = d.read()
    n = a.bits
    res = neg(a)
    zero = const(0, n)
    af = trunc(shr(xor(a, res), const(4, 8)), 1)
    L.setf(("sub", zero, a, res), cf=not_(eq(a, zero)), of=eq(a, const(1 << (n - 1), n)), af=af,
           **L.zsp(res))
    d.write(res)


@handles("NOT")
def _not(L, ins):
    d = L.operand(ins, 0)
    d.write(not_(d.read()))


# ---------------------------------------------------------------------------------------
# shifts and rotates
# ---------------------------------------------------------------------------------------
def _count_operand(L, ins):
    s = L.operand(ins, 1, 8)
    c = s.read()
    return c


@handles("SHL", "SAL", "SHR", "SAR")
def _shift(L, ins):
    d = L.operand(ins, 0)
    cnt = _count_operand(L, ins)
    a = d.read()
    n = a.bits
    kind = {MN.SHL: "shl", MN.SAL: "shl", MN.SHR: "shr", MN.SAR: "sar"}[ins.mnemonic]
    fn = {"shl": shl, "shr": shr, "sar": sar}[kind]
    res = fn(a, cnt)
    if cnt.op == "c":
        k = cnt.args[0] & 31
        if k == 0:
            return
        if kind == "shl":
            cf = bit_at(a, n - k) if k <= n else undef(1)
            of = xor(msb(res), cf) if k == 1 else undef(1)
        elif kind == "shr":
            cf = bit_at(a, k - 1) if k <= n else undef(1)
            of = msb(a) if k == 1 else undef(1)
        else:
            cf = bit_at(a, min(k, n) - 1)
            of = FALSE if k == 1 else undef(1)
        L.setf(("res", res), cf=cf, of=of, af=undef(1), **L.zsp(res))
    else:
        zero = L._zero_count(cnt)
        c31 = and_(cnt, const(31, 8))
        if n == 32:
            if kind == "shl":
                cf = bit_at(a, sub(const(32, 8), c31))
            else:
                cf = bit_at(a, sub(c31, const(1, 8)))
        else:
            cf = undef(1)
        L._keep_if_zero(zero, cf=cf, of=undef(1), af=undef(1), **L.zsp(res))
    d.write(res)


@handles("ROL", "ROR")
def _rotate(L, ins):
    d = L.operand(ins, 0)
    cnt = _count_operand(L, ins)
    a = d.read()
    n = a.bits
    left = ins.mnemonic == MN.ROL
    res = (rol if left else ror)(a, cnt)
    if cnt.op == "c":
        k = cnt.args[0] & 31
        if k == 0:
            return
        cf = lsb(res) if left else msb(res)
        if k == 1:
            of = xor(msb(res), cf) if left else xor(msb(res), bit_at(res, n - 2))
        else:
            of = undef(1)
        L.setf(None, cf=cf, of=of)
    else:
        zero = L._zero_count(cnt)
        L._keep_if_zero(zero, cf=lsb(res) if left else msb(res), of=undef(1))
    d.write(res)


@handles("RCL", "RCR")
def _rotate_carry(L, ins):
    d = L.operand(ins, 0)
    cnt = _count_operand(L, ins)
    a = d.read()
    n = a.bits
    cin = L.st.flags["cf"]
    left = ins.mnemonic == MN.RCL
    if cnt.op != "c":
        res = (T.rcl if left else T.rcr)(a, cnt, cin)
        L.setf(None, cf=undef(1), of=undef(1))
        d.write(res)
        return
    k = cnt.args[0] & 31
    ce = k % (n + 1)
    if k == 0:
        return
    if ce == 0:
        L.setf(None, of=undef(1))
        return
    cz = zext(cin, n)
    if left:
        res = or_(or_(lsh(a, ce), lsh(cz, ce - 1)), rsh(a, n + 1 - ce))
        cf = bit_at(a, n - ce)
        of = xor(msb(res), cf) if k == 1 else undef(1)
    else:
        res = or_(or_(rsh(a, ce), lsh(cz, n - ce)), lsh(a, n + 1 - ce))
        cf = bit_at(a, ce - 1)
        of = xor(msb(res), bit_at(res, n - 2)) if k == 1 else undef(1)
    L.setf(None, cf=cf, of=of)
    d.write(res)


@handles("SHLD", "SHRD")
def _shld(L, ins):
    d = L.operand(ins, 0)
    if d.bits != 32:
        raise Unsupported(ins, "16 bit double shift")
    s = L.operand(ins, 1)
    cnt = L.operand(ins, 2, 8).read()
    dst, src = d.read(), s.read()
    left = ins.mnemonic == MN.SHLD
    res = (T.shld if left else T.shrd)(dst, src, cnt)
    if cnt.op == "c":
        k = cnt.args[0] & 31
        if k == 0:
            return
        cf = bit_at(dst, 32 - k) if left else bit_at(dst, k - 1)
        of = xor(msb(res), msb(dst)) if k == 1 else undef(1)
        L.setf(("res", res), cf=cf, of=of, af=undef(1), **L.zsp(res))
    else:
        zero = L._zero_count(cnt)
        c31 = and_(cnt, const(31, 8))
        cf = bit_at(dst, sub(const(32, 8), c31)) if left else bit_at(dst, sub(c31, const(1, 8)))
        L._keep_if_zero(zero, cf=cf, of=undef(1), af=undef(1), **L.zsp(res))
    d.write(res)


# ---------------------------------------------------------------------------------------
# bit tests
# ---------------------------------------------------------------------------------------
def _bt(kind):
    def fn(L, ins):
        d = L.operand(ins, 0)
        n = d.bits
        if ins.op_kind(1) == OK.REGISTER:
            if isinstance(d, MemOp):
                raise Unsupported(ins, "bit test of memory with register offset")
            off = L.operand(ins, 1).read()
            off = trunc(off, 8) if off.bits > 8 else off
        else:
            off = L.operand(ins, 1, 8).read()
        off = and_(off, const(n - 1, 8))
        a = d.read()
        bitv = trunc(shr(a, off), 1)
        L.setf(None, cf=bitv, of=undef(1), sf=undef(1), af=undef(1), pf=undef(1))
        if kind != "bt":
            m = shl(const(1, n), off)
            res = {"bts": lambda: or_(a, m), "btr": lambda: and_(a, not_(m)), "btc": lambda: xor(a, m)}[kind]()
            d.write(res)
    return fn


for _n in ("bt", "bts", "btr", "btc"):
    HANDLERS[getattr(MN, _n.upper())] = _bt(_n)


@handles("BSF", "BSR")
def _bsx(L, ins):
    d = L.operand(ins, 0)
    s = L.operand(ins, 1)
    src = s.read()
    fn = T.bsf if ins.mnemonic == MN.BSF else T.bsr
    L.setf(None, zf=eq(src, const(0, src.bits)), cf=undef(1), of=undef(1), sf=undef(1), af=undef(1),
           pf=undef(1))
    d.write(fn(src, d.read()))


# ---------------------------------------------------------------------------------------
# multiplication / division
# ---------------------------------------------------------------------------------------
@handles("IMUL")
def _imul(L, ins):
    st = L.st
    if ins.op_count == 1:
        s = L.operand(ins, 0)
        if s.bits != 32:
            raise Unsupported(ins, "8/16 bit one-operand imul")
        a, b = st.regs["eax"], s.read()
        lo, hi = mul(a, b), T.mulhs(a, b)
        st.regs["eax"], st.regs["edx"] = lo, hi
        cfof = not_(eq(hi, sar(lo, const(31, 8))))
    else:
        d = L.operand(ins, 0)
        if ins.op_count == 2:
            a, b = d.read(), L.operand(ins, 1).read()
        else:
            a, b = L.operand(ins, 1).read(), L.operand(ins, 2, d.bits).read()
        n = d.bits
        lo = mul(a, b)
        if n == 32:
            cfof = not_(eq(T.mulhs(a, b), sar(lo, const(31, 8))))
        else:
            p = mul(sext(a, 32), sext(b, 32))
            cfof = not_(eq(sext(lo, 32), p))
        d.write(lo)
    L.setf(None, cf=cfof, of=cfof, zf=undef(1), sf=undef(1), af=undef(1), pf=undef(1))


@handles("MUL")
def _mul(L, ins):
    st = L.st
    s = L.operand(ins, 0)
    if s.bits != 32:
        raise Unsupported(ins, "8/16 bit mul")
    a, b = st.regs["eax"], s.read()
    lo, hi = mul(a, b), T.mulhu(a, b)
    st.regs["eax"], st.regs["edx"] = lo, hi
    cfof = not_(eq(hi, const(0)))
    L.setf(None, cf=cfof, of=cfof, zf=undef(1), sf=undef(1), af=undef(1), pf=undef(1))


@handles("DIV", "IDIV")
def _div(L, ins):
    st = L.st
    s = L.operand(ins, 0)
    if s.bits != 32:
        raise Unsupported(ins, "8/16 bit division")
    d = s.read()
    hi, lo = st.regs["edx"], st.regs["eax"]
    signed = ins.mnemonic == MN.IDIV
    q = T.divmod_op("divs" if signed else "divu", hi, lo, d)
    r = T.divmod_op("rems" if signed else "remu", hi, lo, d)
    st.regs["eax"], st.regs["edx"] = q, r
    L.setf(None, cf=undef(1), of=undef(1), zf=undef(1), sf=undef(1), af=undef(1), pf=undef(1))


@handles("CDQ")
def _cdq(L, ins):
    L.st.regs["edx"] = sar(L.st.regs["eax"], const(31, 8))


@handles("CWDE")
def _cwde(L, ins):
    L.st.regs["eax"] = sext(trunc(L.st.regs["eax"], 16), 32)


@handles("CBW")
def _cbw(L, ins):
    L.st.write_reg(R.AX, sext(L.st.read_reg(R.AL), 16))


@handles("CWD")
def _cwd(L, ins):
    L.st.write_reg(R.DX, sar(L.st.read_reg(R.AX), const(15, 8)))


# ---------------------------------------------------------------------------------------
# conditional data movement
# ---------------------------------------------------------------------------------------
def _cmov(L, ins):
    d = L.operand(ins, 0)
    s = L.operand(ins, 1)
    c = L.cond_for(ins)
    src = s.read()                    # the load happens regardless of the condition
    d.write(ite(c, src, d.read()))


def _setcc(L, ins):
    d = L.operand(ins, 0)
    d.write(zext(L.cond_for(ins), 8))


for _n in ("O", "NO", "B", "AE", "E", "NE", "BE", "A", "S", "NS", "P", "NP", "L", "GE", "LE", "G"):
    HANDLERS[getattr(MN, "CMOV" + _n)] = _cmov
    HANDLERS[getattr(MN, "SET" + _n)] = _setcc


# ---------------------------------------------------------------------------------------
# flag manipulation
# ---------------------------------------------------------------------------------------
@handles("CLC")
def _clc(L, ins):
    L.setf(None, cf=FALSE)


@handles("STC")
def _stc(L, ins):
    L.setf(None, cf=TRUE)


@handles("CMC")
def _cmc(L, ins):
    L.setf(None, cf=not_(L.st.flags["cf"]))


@handles("CLD")
def _cld(L, ins):
    L.setf(None, df=FALSE)


@handles("STD")
def _std(L, ins):
    L.setf(None, df=TRUE)


def _pack_flags(st: State) -> T.Term:
    f = st.flags
    v = const(0x202)
    for name, pos in (("cf", 0), ("pf", 2), ("af", 4), ("zf", 6), ("sf", 7), ("df", 10), ("of", 11)):
        b = zext(f[name], 32)
        v = or_(v, shl(b, const(pos, 8)) if pos else b)
    return v


def _unpack_flags(L: Lifter, v: T.Term):
    L.setf(None, **{n: bit_at(v, pos) for n, pos in
                    (("cf", 0), ("pf", 2), ("af", 4), ("zf", 6), ("sf", 7), ("df", 10), ("of", 11))})


@handles("PUSHFD")
def _pushfd(L, ins):
    L.push_val(_pack_flags(L.st))


@handles("POPFD")
def _popfd(L, ins):
    _unpack_flags(L, L.pop_val())


@handles("LAHF")
def _lahf(L, ins):
    f = L.st.flags
    v = const(0x02, 8)
    for name, pos in (("cf", 0), ("pf", 2), ("af", 4), ("zf", 6), ("sf", 7)):
        b = zext(f[name], 8)
        v = or_(v, shl(b, const(pos, 8)) if pos else b)
    L.st.write_reg(R.AH, v)


@handles("SAHF")
def _sahf(L, ins):
    ah = L.st.read_reg(R.AH)
    L.setf(None, **{n: bit_at(ah, pos) for n, pos in (("cf", 0), ("pf", 2), ("af", 4), ("zf", 6), ("sf", 7))})


# ---------------------------------------------------------------------------------------
# string stores
# ---------------------------------------------------------------------------------------
REP_LIMIT = 4096      # unrolled repetitions per instruction (and the count must be known)
REP_MNEMONICS = frozenset({MN.STOSB, MN.STOSW, MN.STOSD})


@handles("STOSB", "STOSW", "STOSD")
def _stos(L, ins):
    """``rep stos`` is a loop, so it is unrolled - which needs a constant count."""
    st = L.st
    nbytes = {MN.STOSB: 1, MN.STOSW: 2, MN.STOSD: 4}[ins.mnemonic]
    if st.flags["df"] != FALSE:
        raise Unsupported(ins, "direction flag set")
    if nbytes == 4:
        val = st.regs["eax"]
    elif nbytes == 2:
        val = st.read_reg(R.AX)
    else:
        val = st.read_reg(R.AL)
    count = 1
    if ins.has_rep_prefix or ins.has_repe_prefix or ins.has_repne_prefix:
        c = st.regs["ecx"]
        if c.op != "c":
            raise Unsupported(ins, "rep with a symbolic count")
        count = c.args[0]
        st.regs["ecx"] = const(0)
    if count > REP_LIMIT:
        raise Unsupported(ins, f"rep of {count} iterations")
    addr = st.regs["edi"]
    for i in range(count):
        st.mem.store(None, add(addr, const(nbytes * i)), nbytes, val)
    if count:
        st.regs["edi"] = add(addr, const(nbytes * count))


# ---------------------------------------------------------------------------------------
# stack
# ---------------------------------------------------------------------------------------
@handles("PUSH")
def _push(L, ins):
    if ins.stack_pointer_increment != -4:
        raise Unsupported(ins, "16 bit push / segment register push")
    if ins.op_kind(0) == OK.REGISTER and ins.op_register(0) in SEGINFO:
        L.push_val(T.undef(32))    # a selector: unknown, and never assumed to be anything
        return
    s = L.operand(ins, 0, 32)
    if s.bits != 32:
        raise Unsupported(ins, "16 bit push")
    L.push_val(s.read())


@handles("POP")
def _pop(L, ins):
    if ins.stack_pointer_increment != 4:
        raise Unsupported(ins, "16 bit pop / segment register")
    if ins.op_kind(0) == OK.REGISTER and ins.op_register(0) in SEGINFO:
        L.pop_val()                # writing a selector changes nothing that is modelled
        return
    if ins.op_kind(0) == OK.REGISTER and REGINFO.get(ins.op_register(0), (0, 0))[1] != 32:
        raise Unsupported(ins, "16 bit pop / segment register")
    v = L.pop_val()
    d = L.operand(ins, 0)          # a memory destination uses the ESP *after* the increment
    if d.bits != 32:
        raise Unsupported(ins, "16 bit pop")
    d.write(v)


@handles("PUSHAD")
def _pushad(L, ins):
    st = L.st
    old_esp = st.esp
    for n in ("eax", "ecx", "edx", "ebx"):
        L.push_val(st.regs[n])
    L.push_val(old_esp)
    for n in ("ebp", "esi", "edi"):
        L.push_val(st.regs[n])


@handles("POPAD")
def _popad(L, ins):
    st = L.st
    for n in ("edi", "esi", "ebp"):
        st.regs[n] = L.pop_val()
    st.regs["esp"] = add(st.esp, const(4))      # the saved ESP is discarded
    for n in ("ebx", "edx", "ecx", "eax"):
        st.regs[n] = L.pop_val()


@handles("LEAVE")
def _leave(L, ins):
    if ins.code != Code.LEAVED:
        raise Unsupported(ins, "16 bit leave")
    st = L.st
    st.regs["esp"] = st.regs["ebp"]
    st.regs["ebp"] = L.pop_val()


# ---------------------------------------------------------------------------------------
# opaque effects
# ---------------------------------------------------------------------------------------
@handles("CPUID")
def _cpuid(L, ins):
    st = L.st
    e = T.effect("cpuid", (st.regs["eax"], st.regs["ecx"]))
    for i, n in enumerate(("eax", "ebx", "ecx", "edx")):
        st.regs[n] = T.proj(e, i)


@handles("RDTSC")
def _rdtsc(L, ins):
    st = L.st
    e = T.effect("rdtsc", (), unique=True)
    st.regs["eax"], st.regs["edx"] = T.proj(e, 0), T.proj(e, 1)


# ---------------------------------------------------------------------------------------
# control flow
# ---------------------------------------------------------------------------------------
def _target_flow(term: T.Term) -> Flow:
    if term.op == "c":
        return jump(term.args[0])
    return Flow("indirect", term=term)


@handles("JMP")
def _jmp(L, ins):
    if ins.op_kind(0) == OK.NEAR_BRANCH32:
        return jump(ins.near_branch32)
    return _target_flow(L.operand(ins, 0).read())


@handles("CALL")
def _call(L, ins):
    if ins.op_kind(0) == OK.NEAR_BRANCH32:
        target = const(ins.near_branch32)
    else:
        target = L.operand(ins, 0).read()
    L.push_val(const(ins.next_ip))
    return _target_flow(target)


@handles("RET")
def _ret(L, ins):
    t = L.pop_val()
    if ins.op_count == 1:
        L.st.regs["esp"] = add(L.st.esp, const(ins.immediate(0) & 0xFFFF))
    return _target_flow(t)


def _jcc(L, ins):
    c = L.cond_for(ins)
    taken, fall = ins.near_branch32, ins.next_ip
    if c.op == "c":
        return jump(taken if c.args[0] else fall)
    return Flow("branch", cond=c, taken=taken, fall=fall)


for _n in ("O", "NO", "B", "AE", "E", "NE", "BE", "A", "S", "NS", "P", "NP", "L", "GE", "LE", "G"):
    HANDLERS[getattr(MN, "J" + _n)] = _jcc


@handles("JECXZ")
def _jecxz(L, ins):
    c = eq(L.st.regs["ecx"], const(0))
    taken, fall = ins.near_branch32, ins.next_ip
    if c.op == "c":
        return jump(taken if c.args[0] else fall)
    return Flow("branch", cond=c, taken=taken, fall=fall)


@handles("HLT", "INT3", "UD2", "INT", "INTO", "INT1", "SYSENTER", "SYSCALL")
def _halt(L, ins):
    return Flow("halt")
