"""Hash-consed expression DAG with aggressive local simplification.

Every value the symbolic executor computes (register contents, flag bits,
memory loads, ...) is a ``Term``.  Terms are immutable and interned, so two
structurally identical expressions are the *same object* (free CSE) and a term's
``uid`` doubles as a birth order (operands always have a smaller uid than their
users, effects get uids from the same counter).

Widths are 1 (booleans / flag bits), 8, 16 and 32.  Shift/rotate counts follow
the x86 rule "count & 31"; the count operand is always an 8 bit term.

The module-level constructors (``add``, ``xor``, ``shl`` ...) perform constant
folding and algebraic rewriting; set ``SIMPLIFY = False`` to get a plain,
unsimplified DAG (used by the property tests to validate every rewrite rule).
"""
from __future__ import annotations

import itertools
from typing import Callable, Iterable

MASK = {1: 0x1, 8: 0xFF, 16: 0xFFFF, 32: 0xFFFFFFFF}
SIMPLIFY = True

_counter = itertools.count(1)
_table: dict = {}


def next_uid() -> int:
    return next(_counter)


def reset() -> None:
    """Forget all interned terms (tests only)."""
    _table.clear()


def sx(v: int, bits: int) -> int:
    """Interpret ``v`` as a signed ``bits`` wide integer."""
    return v - (1 << bits) if v >> (bits - 1) else v


def _fill(m: int) -> int:
    """Smallest all-ones mask covering every bit of ``m``."""
    return (1 << m.bit_length()) - 1


class Term:
    __slots__ = ("op", "args", "bits", "uid", "nz", "hint", "__weakref__")

    def __init__(self, op: str, args: tuple, bits: int, nz: int):
        self.op = op
        self.args = args
        self.bits = bits
        self.uid = next_uid()
        self.nz = nz          # mask of bits that can possibly be non-zero
        self.hint = None      # architectural register that held it at birth

    # ---- helpers -------------------------------------------------------
    @property
    def is_const(self) -> bool:
        return self.op == "c"

    @property
    def val(self) -> int:
        assert self.op == "c"
        return self.args[0]

    def operands(self) -> Iterable["Term"]:
        return (a for a in self.args if isinstance(a, Term))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return fmt(self)

    def __lt__(self, other: "Term") -> bool:
        return self.uid < other.uid


def fmt(t: Term, depth: int = 4) -> str:
    if t.op == "c":
        return f"{t.args[0]:#x}" if t.args[0] > 9 else str(t.args[0])
    if t.op in ("in", "undef"):
        return f"{t.op}:{t.args[0]}" if t.op == "in" else f"undef{t.args[0]}"
    if depth <= 0:
        return f"#{t.uid}"
    inner = ", ".join(fmt(a, depth - 1) if isinstance(a, Term) else repr(a) for a in t.args)
    return f"{t.op}{t.bits if t.bits not in (32,) else ''}({inner})"


def mk(op: str, args: tuple, bits: int, nz: int | None = None) -> Term:
    key = (op, bits, tuple(a.uid if isinstance(a, Term) else a for a in args))
    t = _table.get(key)
    if t is None:
        if nz is None:
            nz = _compute_nz(op, args, bits)
        t = _table[key] = Term(op, args, bits, nz)
    return t


def _compute_nz(op: str, args: tuple, bits: int) -> int:
    if bits == 0:                                  # opaque effect handle (cpuid, rdtsc)
        return 0
    M = MASK[bits]
    if op == "c":
        return args[0]
    if op in ("eq", "ult", "slt", "par"):
        return 1
    if op == "zext":
        return args[0].nz
    if op == "sext":
        x = args[0]
        return x.nz if not (x.nz >> (x.bits - 1)) else M
    if op == "trunc":
        return args[0].nz & M
    if op == "and":
        return args[0].nz & args[1].nz
    if op in ("or", "xor"):
        return args[0].nz | args[1].nz
    if op == "shl":
        c = args[1]
        if c.op == "c":
            return (args[0].nz << (c.args[0] & 31)) & M
        return M
    if op == "shr":
        c = args[1]
        if c.op == "c":
            return args[0].nz >> (c.args[0] & 31)
        return _fill(args[0].nz)
    if op == "sar":
        x = args[0]
        if not (x.nz >> (x.bits - 1)):
            return _fill(x.nz) if args[1].op != "c" else x.nz >> (args[1].args[0] & 31)
        return M
    if op == "add":
        return ((1 << (max(args[0].nz.bit_length(), args[1].nz.bit_length()) + 1)) - 1) & M
    if op == "mul":
        return ((1 << (args[0].nz.bit_length() + args[1].nz.bit_length())) - 1) & M
    if op == "ite":
        return args[1].nz | args[2].nz
    if op == "ins":
        base, part, shift = args
        pm = MASK[part.bits] << shift
        return ((base.nz & ~pm) | (part.nz << shift)) & M
    return M


# --------------------------------------------------------------------------
# leaves
# --------------------------------------------------------------------------
def const(v: int, bits: int = 32) -> Term:
    return mk("c", (v & MASK[bits],), bits)


def inp(name: str, bits: int = 32) -> Term:
    return mk("in", (name,), bits)


_undef_counter = itertools.count(1)


def undef(bits: int = 32) -> Term:
    return mk("undef", (next(_undef_counter),), bits)


TRUE = const(1, 1)
FALSE = const(0, 1)


def is_c(t: Term) -> bool:
    return t.op == "c"


def cv(t: Term) -> int:
    return t.args[0]


# --------------------------------------------------------------------------
# concrete semantics (shared by the constant folder and the evaluator)
# --------------------------------------------------------------------------
def _parity(x: int) -> int:
    return 1 if bin(x & 0xFF).count("1") % 2 == 0 else 0


def _rcl(a, cnt, cin, n):
    ce = (cnt & 31) % (n + 1)
    if ce == 0:
        return a
    wide = (cin << n) | a
    w = n + 1
    wide = ((wide << ce) | (wide >> (w - ce))) & ((1 << w) - 1)
    return wide & ((1 << n) - 1)


def _rcr(a, cnt, cin, n):
    ce = (cnt & 31) % (n + 1)
    if ce == 0:
        return a
    wide = (cin << n) | a
    w = n + 1
    wide = ((wide >> ce) | (wide << (w - ce))) & ((1 << w) - 1)
    return wide & ((1 << n) - 1)


def apply_op(op: str, bits: int, args: tuple, v: list):
    """Evaluate one node given the integer values ``v`` of its term arguments
    (non-term arguments are passed through unchanged in ``v``)."""
    M = MASK[bits]
    if op == "add":
        return (v[0] + v[1]) & M
    if op == "sub":
        return (v[0] - v[1]) & M
    if op == "mul":
        return (v[0] * v[1]) & M
    if op == "and":
        return v[0] & v[1]
    if op == "or":
        return v[0] | v[1]
    if op == "xor":
        return v[0] ^ v[1]
    if op == "not":
        return ~v[0] & M
    if op == "neg":
        return -v[0] & M
    if op == "shl":
        return (v[0] << (v[1] & 31)) & M
    if op == "shr":
        return v[0] >> (v[1] & 31)
    if op == "sar":
        return (sx(v[0], bits) >> (v[1] & 31)) & M
    if op == "rol":
        c = (v[1] & 31) % bits
        return ((v[0] << c) | (v[0] >> (bits - c))) & M
    if op == "ror":
        c = (v[1] & 31) % bits
        return ((v[0] >> c) | (v[0] << (bits - c))) & M
    if op == "bswap":
        return int.from_bytes(v[0].to_bytes(4, "little"), "big") if bits == 32 else 0
    if op == "zext":
        return v[0]
    if op == "sext":
        return sx(v[0], args[0].bits) & M
    if op == "trunc":
        return v[0] & M
    if op == "ins":
        pm = MASK[args[1].bits] << v[2]
        return (v[0] & ~pm & M) | (v[1] << v[2])
    if op == "ite":
        return v[1] if v[0] else v[2]
    if op == "eq":
        return int(v[0] == v[1])
    if op == "ult":
        return int(v[0] < v[1])
    if op == "slt":
        b = args[0].bits
        return int(sx(v[0], b) < sx(v[1], b))
    if op == "par":
        return _parity(v[0])
    if op == "mulhu":
        return (v[0] * v[1]) >> 32
    if op == "mulhs":
        return ((sx(v[0], 32) * sx(v[1], 32)) >> 32) & M
    if op in ("divu", "remu"):
        d = v[2]
        if d == 0:
            return 0
        n = (v[0] << 32) | v[1]
        return (n // d if op == "divu" else n % d) & M
    if op in ("divs", "rems"):
        d = sx(v[2], 32)
        if d == 0:
            return 0
        n = sx(((v[0] << 32) | v[1]), 64)
        q = abs(n) // abs(d)
        if (n < 0) != (d < 0):
            q = -q
        r = n - q * d
        return (q if op == "divs" else r) & M
    if op == "bsf":
        return (v[0] & -v[0]).bit_length() - 1 if v[0] else v[1]
    if op == "bsr":
        return v[0].bit_length() - 1 if v[0] else v[1]
    if op == "shld":
        c = v[2] & 31
        return v[0] if c == 0 else ((v[0] << c) | (v[1] >> (32 - c))) & M
    if op == "shrd":
        c = v[2] & 31
        return v[0] if c == 0 else ((v[0] >> c) | (v[1] << (32 - c))) & M
    if op == "rcl":
        return _rcl(v[0], v[1], v[2], bits)
    if op == "rcr":
        return _rcr(v[0], v[1], v[2], bits)
    raise KeyError(op)


EXTERNAL_OPS = {"in", "undef", "load", "eff", "proj", "c"}


# --------------------------------------------------------------------------
# smart constructors
# --------------------------------------------------------------------------
def _fold(op: str, bits: int, *ts: Term):
    """Constant-fold when every term argument is a constant."""
    if all(t.op == "c" for t in ts):
        v = [t.args[0] for t in ts]
        return const(apply_op(op, bits, tuple(ts), v), bits)
    return None


def _order(a: Term, b: Term):
    """Canonical operand order of a commutative operation: const last."""
    if a.op == "c" and b.op != "c":
        return b, a
    if a.op != "c" and b.op != "c" and a.uid > b.uid:
        return b, a
    return a, b


def add(a: Term, b: Term) -> Term:
    bits = a.bits
    assert b.bits == bits, (a, b)
    if SIMPLIFY:
        f = _fold("add", bits, a, b)
        if f is not None:
            return f
        a, b = _order(a, b)
        if b.op == "c":
            c = b.args[0]
            if c == 0:
                return a
            if a.op == "add" and a.args[1].op == "c":
                return add(a.args[0], const(a.args[1].args[0] + c, bits))
        else:
            # x + (y - x) -> y ; (y - x) + x -> y
            for p, q in ((a, b), (b, a)):
                if q.op == "sub" and q.args[1] is p:
                    return q.args[0]
            if a.op == "neg":
                return sub(b, a.args[0])
            if b.op == "neg":
                return sub(a, b.args[0])
    return mk("add", (a, b), bits)


def sub(a: Term, b: Term) -> Term:
    bits = a.bits
    assert b.bits == bits, (a, b)
    if SIMPLIFY:
        f = _fold("sub", bits, a, b)
        if f is not None:
            return f
        if a is b:
            return const(0, bits)
        if b.op == "c":
            return add(a, const(-b.args[0], bits))
        if a.op == "c" and a.args[0] == 0:
            return neg(b)
        if a.op == "add":
            if a.args[1] is b:
                return a.args[0]
            if a.args[0] is b:
                return a.args[1]
        if b.op == "add":
            if b.args[0] is a:
                return neg(b.args[1])
            if b.args[1] is a:
                return neg(b.args[0])
        if b.op == "neg":
            return add(a, b.args[0])
    return mk("sub", (a, b), bits)


def neg(a: Term) -> Term:
    bits = a.bits
    if SIMPLIFY:
        f = _fold("neg", bits, a)
        if f is not None:
            return f
        if a.op == "neg":
            return a.args[0]
        if a.op == "sub":
            return sub(a.args[1], a.args[0])
        if a.op == "not":
            return add(a.args[0], const(1, bits))
    return mk("neg", (a,), bits)


def mul(a: Term, b: Term) -> Term:
    bits = a.bits
    assert b.bits == bits
    if SIMPLIFY:
        f = _fold("mul", bits, a, b)
        if f is not None:
            return f
        a, b = _order(a, b)
        if b.op == "c":
            c = b.args[0]
            if c == 0:
                return const(0, bits)
            if c == 1:
                return a
            if c & (c - 1) == 0:
                return shl(a, const(c.bit_length() - 1, 8))
            if a.op == "mul" and a.args[1].op == "c":
                return mul(a.args[0], const(a.args[1].args[0] * c, bits))
            if a.op == "add" and a.args[1].op == "c":
                return add(mul(a.args[0], b), const(a.args[1].args[0] * c, bits))
    return mk("mul", (a, b), bits)


def _is_allones(t: Term) -> bool:
    return t.op == "c" and t.args[0] == MASK[t.bits]


def and_(a: Term, b: Term) -> Term:
    bits = a.bits
    assert b.bits == bits, (a, b)
    if SIMPLIFY:
        f = _fold("and", bits, a, b)
        if f is not None:
            return f
        a, b = _order(a, b)
        if a is b:
            return a
        if b.op == "c":
            c = b.args[0]
            if c == 0 or (a.nz & c) == 0:
                return const(0, bits)
            if (a.nz & ~c) == 0:
                return a
            if a.op == "and" and a.args[1].op == "c":
                return and_(a.args[0], const(a.args[1].args[0] & c, bits))
            if bits == 32 and c in (0xFF, 0xFFFF) and a.op != "zext":
                n = 8 if c == 0xFF else 16
                return zext(trunc(a, n), 32)
            if a.op == "zext" and a.args[0].bits <= 16 and c < (1 << a.args[0].bits) - 1 + 1:
                pass
        if a.op == "not" and a.args[0] is b or b.op == "not" and b.args[0] is a:
            return const(0, bits)
    return mk("and", (a, b), bits)


def or_(a: Term, b: Term) -> Term:
    bits = a.bits
    assert b.bits == bits, (a, b)
    if SIMPLIFY:
        f = _fold("or", bits, a, b)
        if f is not None:
            return f
        a, b = _order(a, b)
        if a is b:
            return a
        if b.op == "c":
            c = b.args[0]
            if c == 0:
                return a
            if c == MASK[bits]:
                return b
            if a.op == "or" and a.args[1].op == "c":
                return or_(a.args[0], const(a.args[1].args[0] | c, bits))
            if (a.nz & ~c) == 0:
                return b
        if a.op == "not" and a.args[0] is b or b.op == "not" and b.args[0] is a:
            return const(MASK[bits], bits)
    return mk("or", (a, b), bits)


def xor(a: Term, b: Term) -> Term:
    bits = a.bits
    assert b.bits == bits, (a, b)
    if SIMPLIFY:
        f = _fold("xor", bits, a, b)
        if f is not None:
            return f
        a, b = _order(a, b)
        if a is b:
            return const(0, bits)
        if b.op == "c":
            c = b.args[0]
            if c == 0:
                return a
            if c == MASK[bits]:
                return not_(a)
            if a.op == "xor" and a.args[1].op == "c":
                return xor(a.args[0], const(a.args[1].args[0] ^ c, bits))
            if a.op == "not":
                return xor(a.args[0], const(~c, bits))
        if a.op == "not" and a.args[0] is b or b.op == "not" and b.args[0] is a:
            return const(MASK[bits], bits)
    return mk("xor", (a, b), bits)


def not_(a: Term) -> Term:
    bits = a.bits
    if SIMPLIFY:
        f = _fold("not", bits, a)
        if f is not None:
            return f
        if a.op == "not":
            return a.args[0]
        if a.op == "xor" and a.args[1].op == "c":
            return xor(a.args[0], const(~a.args[1].args[0], bits))
        if a.op == "neg":
            return add(a.args[0], const(-1, bits))
        if bits == 1:
            # boolean negation of comparisons
            if a.op == "ite" and a.args[1].op == "c" and a.args[2].op == "c":
                return ite(a.args[0], a.args[2], a.args[1])
    return mk("not", (a,), bits)


def _count(c: Term) -> Term:
    """Normalise a shift count operand to an 8 bit term."""
    if c.bits != 8:
        c = trunc(c, 8) if c.bits > 8 else zext(c, 8)
    if SIMPLIFY and c.op == "and" and c.args[1].op == "c" and (c.args[1].args[0] & 31) == 31:
        return c.args[0]
    return c


def _shift(op: str, a: Term, c: Term) -> Term:
    bits = a.bits
    c = _count(c)
    if not SIMPLIFY:
        return mk(op, (a, c), bits)
    if c.op == "c":
        k = c.args[0] & 31
        c = const(k, 8)
        if k == 0:
            return a
        if a.op == "c":
            return const(apply_op(op, bits, (a, c), [a.args[0], k]), bits)
        if k >= bits:
            if op == "sar":
                c = const(bits - 1, 8)
                k = bits - 1
            else:
                return const(0, bits)
        if op == "shl":
            if a.op == "shl" and a.args[1].op == "c":
                k2 = a.args[1].args[0] + k
                return const(0, bits) if k2 >= bits else shl(a.args[0], const(k2, 8))
            if a.op == "shr" and a.args[1].op == "c" and a.args[1].args[0] == k:
                return and_(a.args[0], const((MASK[bits] << k), bits))
            if a.op == "add" and a.args[1].op == "c":
                return add(shl(a.args[0], c), const(a.args[1].args[0] << k, bits))
            if a.op == "and" and a.args[1].op == "c":
                return and_(shl(a.args[0], c), const(a.args[1].args[0] << k, bits))
            if (a.nz << k) & MASK[bits] == 0:
                return const(0, bits)
        elif op == "shr":
            if a.op == "shr" and a.args[1].op == "c":
                k2 = a.args[1].args[0] + k
                return const(0, bits) if k2 >= bits else shr(a.args[0], const(k2, 8))
            if a.op == "shl" and a.args[1].op == "c" and a.args[1].args[0] == k:
                return and_(a.args[0], const(MASK[bits] >> k, bits))
            if (a.nz >> k) == 0:
                return const(0, bits)
        elif op == "sar":
            if a.nz >> (bits - 1) == 0:
                return shr(a, c)
            if a.op == "sar" and a.args[1].op == "c":
                return sar(a.args[0], const(min(a.args[1].args[0] + k, bits - 1), 8))
    return mk(op, (a, c), bits)


def shl(a: Term, c: Term) -> Term:
    return _shift("shl", a, c)


def shr(a: Term, c: Term) -> Term:
    return _shift("shr", a, c)


def sar(a: Term, c: Term) -> Term:
    return _shift("sar", a, c)


def _rot(op: str, a: Term, c: Term) -> Term:
    bits = a.bits
    c = _count(c)
    if not SIMPLIFY:
        return mk(op, (a, c), bits)
    if c.op == "c":
        k = (c.args[0] & 31) % bits
        if k == 0:
            return a
        if a.op == "c":
            return const(apply_op(op, bits, (a, c), [a.args[0], k]), bits)
        if op == "ror":
            op, k = "rol", bits - k
        c = const(k, 8)
        if a.op == "rol" and a.args[1].op == "c":
            return rol(a.args[0], const(a.args[1].args[0] + k, 8))
        if a.op == "bswap" and bits == 32 and k == 16:
            pass
        if (a.nz == 0):
            return a
    return mk(op, (a, c), bits)


def rol(a: Term, c: Term) -> Term:
    return _rot("rol", a, c)


def ror(a: Term, c: Term) -> Term:
    return _rot("ror", a, c)


def bswap(a: Term) -> Term:
    if SIMPLIFY:
        f = _fold("bswap", a.bits, a)
        if f is not None:
            return f
        if a.op == "bswap":
            return a.args[0]
    return mk("bswap", (a,), a.bits)


def zext(a: Term, bits: int) -> Term:
    if a.bits == bits:
        return a
    assert bits > a.bits
    if SIMPLIFY:
        if a.op == "c":
            return const(a.args[0], bits)
        if a.op == "zext":
            return zext(a.args[0], bits)
        if a.op == "trunc" and a.args[0].bits == bits and (a.args[0].nz & ~MASK[a.bits]) == 0:
            return a.args[0]
        if a.op == "ite" and a.args[1].op == "c" and a.args[2].op == "c" and False:
            pass
    return mk("zext", (a,), bits)


def sext(a: Term, bits: int) -> Term:
    if a.bits == bits:
        return a
    assert bits > a.bits
    if SIMPLIFY:
        if a.op == "c":
            return const(sx(a.args[0], a.bits), bits)
        if a.op == "sext":
            return sext(a.args[0], bits)
        if not (a.nz >> (a.bits - 1)):
            return zext(a, bits)
    return mk("sext", (a,), bits)


_RING = {"add", "sub", "mul", "and", "or", "xor"}


def trunc(a: Term, bits: int) -> Term:
    if a.bits == bits:
        return a
    assert bits < a.bits, (a, bits)
    if not SIMPLIFY:
        return mk("trunc", (a,), bits)
    op = a.op
    if op == "c":
        return const(a.args[0], bits)
    if op == "trunc":
        return trunc(a.args[0], bits)
    if op in ("zext", "sext"):
        y = a.args[0]
        if y.bits == bits:
            return y
        if y.bits > bits:
            return trunc(y, bits)
        return (zext if op == "zext" else sext)(y, bits)
    if op in _RING:
        return {"add": add, "sub": sub, "mul": mul, "and": and_, "or": or_, "xor": xor}[op](
            trunc(a.args[0], bits), trunc(a.args[1], bits))
    if op == "not":
        return not_(trunc(a.args[0], bits))
    if op == "neg":
        return neg(trunc(a.args[0], bits))
    if op == "shl":
        return shl(trunc(a.args[0], bits), a.args[1])
    if op == "shr" and a.args[1].op == "c":
        # reading a field back out of an insert - how a high byte register (ah,
        # ch, ...) is read: the bits come from the inserted part, or miss it
        x, cnt = a.args
        k = cnt.args[0] & 31
        if x.op == "ins":
            base, part, sh = x.args
            if sh <= k and k + bits <= sh + part.bits:
                return trunc(shr(part, const(k - sh, 8)), bits)
            if k + bits <= sh or k >= sh + part.bits:
                return trunc(shr(base, cnt), bits)
    if op == "ite":
        return ite(a.args[0], trunc(a.args[1], bits), trunc(a.args[2], bits))
    if op == "ins":
        base, part, sh = a.args
        if sh == 0 and part.bits >= bits:
            return trunc(part, bits)
        if sh >= bits:
            return trunc(base, bits)
        if sh + part.bits <= bits:
            return ins(trunc(base, bits), part, sh)
    if op == "load" and bits >= 8:       # little endian: the low bytes live at the same address
        seg, addr, mstate = a.args
        return load(seg, addr, mstate, bits)
    if a.nz & MASK[bits] == 0:
        return const(0, bits)
    return mk("trunc", (a,), bits)


def ins(base: Term, part: Term, shift: int) -> Term:
    """Replace ``part.bits`` bits of ``base`` at bit offset ``shift``."""
    if part.bits == base.bits:
        return part
    if not SIMPLIFY:
        return mk("ins", (base, part, shift), base.bits)
    if base.op == "c" and part.op == "c":
        pm = MASK[part.bits] << shift
        return const((base.args[0] & ~pm) | (part.args[0] << shift), base.bits)
    # writing back the very bits we already have
    if shift == 0 and part.op == "trunc" and part.args[0] is base:
        return base
    if (shift == 8 and part.op == "trunc" and part.args[0].op == "shr"
            and part.args[0].args[0] is base and part.args[0].args[1].op == "c"
            and part.args[0].args[1].args[0] == 8 and part.bits == 8):
        return base
    # overwrite of an earlier insert at the same place
    if base.op == "ins" and base.args[2] == shift and base.args[1].bits == part.bits:
        return ins(base.args[0], part, shift)
    pm = MASK[part.bits] << shift
    if (base.nz & ~pm) == 0:
        z = zext(part, base.bits)
        return z if shift == 0 else shl(z, const(shift, 8))
    return mk("ins", (base, part, shift), base.bits)


def ite(c: Term, a: Term, b: Term) -> Term:
    assert c.bits == 1 and a.bits == b.bits, (c, a, b)
    if SIMPLIFY:
        if c.op == "c":
            return a if c.args[0] else b
        if a is b:
            return a
        if c.op == "not":
            return ite(c.args[0], b, a)
        if a.bits == 1 and a.op == "c" and b.op == "c":
            return c if a.args[0] == 1 else not_(c)
    return mk("ite", (c, a, b), a.bits)


def eq(a: Term, b: Term) -> Term:
    assert a.bits == b.bits, (a, b)
    if SIMPLIFY:
        f = _fold("eq", 1, a, b)
        if f is not None:
            return f
        if a is b:
            return TRUE
        a, b = _order(a, b)
        if b.op == "c":
            c = b.args[0]
            if (a.nz & c) != c:
                return FALSE
            if a.bits == 1:
                return a if c == 1 else not_(a)
            if c == 0:
                if a.op == "sub":
                    return eq(a.args[0], a.args[1])
                if a.op == "xor":
                    return eq(a.args[0], a.args[1])
                if a.op == "neg" or a.op == "not" and False:
                    return eq(a.args[0], b)
                if a.op == "zext":
                    return eq(a.args[0], const(0, a.args[0].bits))
                if a.op == "shl" and a.args[1].op == "c" and (a.args[0].nz << a.args[1].args[0]) & MASK[a.bits] == (a.args[0].nz << a.args[1].args[0]) & MASK[a.bits]:
                    pass
            if a.op == "add" and a.args[1].op == "c":
                return eq(a.args[0], const(c - a.args[1].args[0], a.bits))
            if a.op == "xor" and a.args[1].op == "c":
                return eq(a.args[0], const(c ^ a.args[1].args[0], a.bits))
            if a.op == "not":
                return eq(a.args[0], const(~c, a.bits))
            if a.op == "zext" and c <= MASK[a.args[0].bits]:
                return eq(a.args[0], const(c, a.args[0].bits))
    return mk("eq", (a, b), 1)


def ult(a: Term, b: Term) -> Term:
    assert a.bits == b.bits, (a, b)
    if SIMPLIFY:
        f = _fold("ult", 1, a, b)
        if f is not None:
            return f
        if a is b:
            return FALSE
        if b.op == "c":
            if b.args[0] == 0:
                return FALSE
            if b.args[0] == 1:
                return eq(a, const(0, a.bits))
            if a.nz < b.args[0]:
                return TRUE
        if a.op == "c":
            if a.args[0] == 0:
                return not_(eq(b, const(0, b.bits)))
            if a.args[0] >= b.nz:
                return FALSE
    return mk("ult", (a, b), 1)


def slt(a: Term, b: Term) -> Term:
    assert a.bits == b.bits, (a, b)
    if SIMPLIFY:
        f = _fold("slt", 1, a, b)
        if f is not None:
            return f
        if a is b:
            return FALSE
        sign = 1 << (a.bits - 1)
        if b.op == "c" and b.args[0] == 0 and a.nz < sign:
            return FALSE
        if a.nz < sign and b.nz < sign:
            return ult(a, b)          # both known non-negative
    return mk("slt", (a, b), 1)


def par(a: Term) -> Term:
    if SIMPLIFY:
        f = _fold("par", 1, a)
        if f is not None:
            return f
    return mk("par", (a,), 1)


def mulhu(a: Term, b: Term) -> Term:
    if SIMPLIFY:
        f = _fold("mulhu", 32, a, b)
        if f is not None:
            return f
        a, b = _order(a, b)
        if (a.nz.bit_length() + b.nz.bit_length()) <= 32:
            return const(0, 32)
    return mk("mulhu", (a, b), 32)


def mulhs(a: Term, b: Term) -> Term:
    if SIMPLIFY:
        f = _fold("mulhs", 32, a, b)
        if f is not None:
            return f
        a, b = _order(a, b)
    return mk("mulhs", (a, b), 32)


def divmod_op(op: str, hi: Term, lo: Term, d: Term) -> Term:
    if SIMPLIFY:
        f = _fold(op, 32, hi, lo, d)
        if f is not None:
            return f
    return mk(op, (hi, lo, d), 32)


def bsf(src: Term, old: Term) -> Term:
    if SIMPLIFY:
        f = _fold("bsf", 32, src, old)
        if f is not None:
            return f
        if src.op == "c" and src.args[0] == 0:
            return old
    return mk("bsf", (src, old), src.bits)


def bsr(src: Term, old: Term) -> Term:
    if SIMPLIFY:
        f = _fold("bsr", 32, src, old)
        if f is not None:
            return f
        if src.op == "c" and src.args[0] == 0:
            return old
    return mk("bsr", (src, old), src.bits)


def shld(dst: Term, src: Term, c: Term) -> Term:
    c = _count(c)
    if SIMPLIFY:
        if c.op == "c":
            k = c.args[0] & 31
            if k == 0:
                return dst
            if dst is src:
                return rol(dst, const(k, 8))
            if dst.op == "c" and src.op == "c":
                return const(apply_op("shld", 32, (dst, src, c), [dst.args[0], src.args[0], k]), 32)
            c = const(k, 8)
    return mk("shld", (dst, src, c), 32)


def shrd(dst: Term, src: Term, c: Term) -> Term:
    c = _count(c)
    if SIMPLIFY:
        if c.op == "c":
            k = c.args[0] & 31
            if k == 0:
                return dst
            if dst is src:
                return ror(dst, const(k, 8))
            if dst.op == "c" and src.op == "c":
                return const(apply_op("shrd", 32, (dst, src, c), [dst.args[0], src.args[0], k]), 32)
            c = const(k, 8)
    return mk("shrd", (dst, src, c), 32)


def rcl(a: Term, c: Term, cin: Term) -> Term:
    c = _count(c)
    if SIMPLIFY and a.op == "c" and c.op == "c" and cin.op == "c":
        return const(apply_op("rcl", a.bits, (a, c, cin), [a.args[0], c.args[0], cin.args[0]]), a.bits)
    return mk("rcl", (a, c, cin), a.bits)


def rcr(a: Term, c: Term, cin: Term) -> Term:
    c = _count(c)
    if SIMPLIFY and a.op == "c" and c.op == "c" and cin.op == "c":
        return const(apply_op("rcr", a.bits, (a, c, cin), [a.args[0], c.args[0], cin.args[0]]), a.bits)
    return mk("rcr", (a, c, cin), a.bits)


def load(seg, addr: Term, mstate: int, bits: int) -> Term:
    """A memory read of ``bits`` bits at ``addr`` as of memory state ``mstate``."""
    return mk("load", (seg, addr, mstate), bits)


_eff_counter = itertools.count(1)


def effect(kind: str, inputs: tuple, unique: bool = False) -> Term:
    """An opaque multi-result instruction (cpuid, rdtsc, ...).  Results are
    read through ``proj``.  ``unique`` effects are never merged."""
    token = next(_eff_counter) if unique else 0
    return mk("eff", (kind, token) + tuple(inputs), 0)


def proj(e: Term, idx: int, bits: int = 32) -> Term:
    return mk("proj", (e, idx), bits)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(t: Term, env: Callable[[Term], int], memo: dict | None = None) -> int:
    """Evaluate ``t``.  ``env`` supplies the value of external terms
    (``in``, ``undef``, ``load``, ``proj``)."""
    if memo is None:
        memo = {}
    stack = [t]
    while stack:
        n = stack[-1]
        if n.uid in memo:
            stack.pop()
            continue
        if n.op == "c":
            memo[n.uid] = n.args[0]
            stack.pop()
            continue
        if n.op in EXTERNAL_OPS:
            # operands of a load / eff may need evaluating first
            pending = [a for a in n.operands() if a.uid not in memo]
            if pending:
                stack.extend(pending)
                continue
            memo[n.uid] = env(n) & MASK[n.bits] if n.bits else 0
            stack.pop()
            continue
        pending = [a for a in n.operands() if a.uid not in memo]
        if pending:
            stack.extend(pending)
            continue
        v = [memo[a.uid] if isinstance(a, Term) else a for a in n.args]
        memo[n.uid] = apply_op(n.op, n.bits, n.args, v)
        stack.pop()
    return memo[t.uid]


def walk(roots: Iterable[Term]):
    """All terms reachable from ``roots`` in birth (uid) order."""
    seen: dict[int, Term] = {}
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n.uid in seen:
            continue
        seen[n.uid] = n
        stack.extend(n.operands())
    return [seen[k] for k in sorted(seen)]
