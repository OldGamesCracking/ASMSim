"""Property test: every simplification rule must preserve the value.

A random expression "recipe" is built twice - once through the simplifying
constructors, once with ``SIMPLIFY = False`` - and both are evaluated on random
inputs.  Constants and inputs are biased towards values that trigger rewrites.
"""
import random

import pytest

from asmsim import terms as T

SPECIAL = [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 0x7F, 0x80, 0xFF, 0x100, 0x7FFF, 0x8000, 0xFFFF,
           0x10000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF, 0x12345678, 0xDEADBEEF]


def rconst(rng, bits):
    if rng.random() < 0.7:
        return rng.choice(SPECIAL) & T.MASK[bits]
    return rng.getrandbits(bits)


def gen(rng, bits, depth):
    """Return a recipe for a term of width ``bits``."""
    if depth <= 0 or rng.random() < 0.18:
        if rng.random() < 0.45:
            return ("c", rconst(rng, bits), bits)
        return ("in", f"v{rng.randrange(3)}_{bits}", bits)
    d = depth - 1
    r = rng.random()
    if bits == 1:
        k = rng.choice(["eq", "ult", "slt", "par", "and", "or", "xor", "not", "ite", "trunc"])
        w = rng.choice([8, 16, 32])
        if k in ("eq", "ult", "slt"):
            return (k, gen(rng, w, d), gen(rng, w, d))
        if k == "par":
            return ("par", gen(rng, w, d))
        if k in ("and", "or", "xor"):
            return (k, gen(rng, 1, d), gen(rng, 1, d))
        if k == "not":
            return ("not", gen(rng, 1, d))
        if k == "ite":
            return ("ite", gen(rng, 1, d), gen(rng, 1, d), gen(rng, 1, d))
        return ("trunc", gen(rng, w, d), 1)
    ops = ["add", "sub", "mul", "and", "or", "xor", "not", "neg", "shl", "shr", "sar", "rol", "ror",
           "ite", "zext", "trunc", "sext", "ins"]
    if bits == 32:
        ops += ["bswap", "mulhu", "mulhs", "bsf", "bsr", "shld", "shrd"]
    k = rng.choice(ops)
    if k in ("add", "sub", "mul", "and", "or", "xor"):
        return (k, gen(rng, bits, d), gen(rng, bits, d))
    if k in ("not", "neg", "bswap"):
        return (k, gen(rng, bits, d))
    if k in ("shl", "shr", "sar", "rol", "ror"):
        cnt = ("c", rng.choice([0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 24, 31, 32, 33, 0xFF]), 8) \
            if rng.random() < 0.6 else gen(rng, 8, d)
        return (k, gen(rng, bits, d), cnt)
    if k in ("shld", "shrd"):
        cnt = ("c", rng.choice([0, 1, 5, 16, 31]), 8) if rng.random() < 0.6 else gen(rng, 8, d)
        return (k, gen(rng, 32, d), gen(rng, 32, d), cnt)
    if k in ("mulhu", "mulhs", "bsf", "bsr"):
        return (k, gen(rng, 32, d), gen(rng, 32, d))
    if k == "ite":
        return ("ite", gen(rng, 1, d), gen(rng, bits, d), gen(rng, bits, d))
    if k == "zext":
        if bits == 8:
            return gen(rng, 8, d)
        return ("zext", gen(rng, rng.choice([b for b in (8, 16, 1) if b < bits]), d), bits)
    if k == "sext":
        if bits == 8:
            return gen(rng, 8, d)
        return ("sext", gen(rng, rng.choice([b for b in (8, 16) if b < bits]), d), bits)
    if k == "trunc":
        if bits == 32:
            return gen(rng, 32, d)
        return ("trunc", gen(rng, rng.choice([b for b in (16, 32) if b > bits]), d), bits)
    if k == "ins":
        if bits == 8:
            return gen(rng, 8, d)
        pb = rng.choice([8, 16]) if bits == 32 else 8
        shifts = [s for s in (0, 8, 16, 24) if s + pb <= bits and (pb == 8 or s % 16 == 0)]
        return ("ins", gen(rng, bits, d), gen(rng, pb, d), rng.choice(shifts))
    raise AssertionError(k)


def build(rec):
    k = rec[0]
    if k == "c":
        return T.const(rec[1], rec[2])
    if k == "in":
        return T.inp(rec[1], rec[2])
    a = [build(x) if isinstance(x, tuple) else x for x in rec[1:]]
    if k in ("zext", "sext", "trunc"):
        return {"zext": T.zext, "sext": T.sext, "trunc": T.trunc}[k](a[0], a[1])
    if k == "ins":
        return T.ins(a[0], a[1], a[2])
    f = {"add": T.add, "sub": T.sub, "mul": T.mul, "and": T.and_, "or": T.or_, "xor": T.xor,
         "not": T.not_, "neg": T.neg, "shl": T.shl, "shr": T.shr, "sar": T.sar, "rol": T.rol,
         "ror": T.ror, "bswap": T.bswap, "ite": T.ite, "eq": T.eq, "ult": T.ult, "slt": T.slt,
         "par": T.par, "mulhu": T.mulhu, "mulhs": T.mulhs, "bsf": T.bsf, "bsr": T.bsr,
         "shld": T.shld, "shrd": T.shrd}[k]
    return f(*a)


def make_env(rng):
    """Input values are a pure function of (seed, name), independent of the
    order in which the evaluator happens to ask for them."""
    seed = rng.getrandbits(32)

    def env(t):
        assert t.op == "in", t
        r = random.Random(f"{seed}:{t.args[0]}")
        return rconst(r, t.bits) if r.random() < 0.5 else r.getrandbits(t.bits)
    return env


@pytest.mark.parametrize("bits", [1, 8, 16, 32])
def test_rewrites_preserve_value(bits):
    rng = random.Random(1234 + bits)
    n_bad = 0
    for i in range(6000):
        rec = gen(rng, bits, rng.randint(1, 6))
        T.SIMPLIFY = False
        try:
            raw = build(rec)
        finally:
            T.SIMPLIFY = True
        simp = build(rec)
        assert raw.bits == simp.bits == bits
        for j in range(5):
            seed = rng.getrandbits(32)
            a = T.evaluate(raw, make_env(random.Random(seed)))
            b = T.evaluate(simp, make_env(random.Random(seed)))   # same seed -> same inputs
            if a != b:
                n_bad += 1
                pytest.fail(f"mismatch for {rec}\n raw={T.fmt(raw, 8)} -> {a:#x}\n simp={T.fmt(simp, 8)} -> {b:#x}")
    assert n_bad == 0
