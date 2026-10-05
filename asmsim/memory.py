"""Symbolic memory: an ordered list of stores plus forwarding / alias reasoning.

The model is deliberately simple and conservative:

* addresses are split into ``base term + constant offset``; two accesses with the
  same base are compared exactly, accesses with different bases *may alias* ...
* ... except for the one assumption that makes stack code analysable: memory
  *below the entry ESP* (the stub's own new stack frame) is private - nobody
  holds a pointer into it - so it never aliases non-stack addresses, and absolute
  (constant) addresses never alias the stack at all.
* a load that cannot be forwarded from a store becomes a ``load`` term tagged
  with the id of the last store that might have affected it.
"""
from __future__ import annotations

from . import terms as T

DISJOINT, COVERS, OVERLAP, UNKNOWN = "disjoint", "covers", "overlap", "unknown"

ESP_IN = None  # set lazily: the interned in:esp term


def esp_term() -> T.Term:
    return T.inp("esp")


class Store:
    __slots__ = ("seg", "addr", "size", "val", "seq", "pinned")

    def __init__(self, seg, addr: T.Term, size: int, val: T.Term):
        self.seg = seg
        self.addr = addr
        self.size = size              # bytes
        self.val = val                # term of 8*size bits
        self.seq = T.next_uid()       # birth order, shares the term uid counter
        self.pinned = False           # some surviving load depends on it

    def __repr__(self):  # pragma: no cover
        return f"Store({self.seg or ''}[{T.fmt(self.addr)}]/{self.size} = {T.fmt(self.val)})"


def split(addr: T.Term):
    """addr -> (base term or None for absolute, signed constant offset)."""
    if addr.op == "c":
        return None, T.sx(addr.args[0], 32)
    if addr.op == "add" and addr.args[1].op == "c":
        return addr.args[0], T.sx(addr.args[1].args[0], 32)
    return addr, 0


def stack_offset(addr: T.Term):
    """Offset relative to the entry ESP if ``addr`` is ESP-derived, else None."""
    base, off = split(addr)
    if base is not None and base.op == "in" and base.args[0] == "esp":
        return off
    return None


def _is_stack(base) -> bool:
    return base is not None and base.op == "in" and base.args[0] == "esp"


DECL_SIZE = 4       # a declared stack argument / memory value is one dword


class Memory:
    def __init__(self, isolate_new_stack: bool = True):
        self.stores: list[Store] = []
        self.isolate_new_stack = isolate_new_stack
        # declared stack arguments: byte offset from the entry ESP -> the term the
        # slot holds (the argument's input term, or a constant when one was given)
        self.arg_slots: dict[int, T.Term] = {}
        # declared memory contents: absolute address -> the 32 bit term stored there
        self.mem_slots: dict[int, T.Term] = {}

    def copy(self) -> "Memory":
        m = Memory(self.isolate_new_stack)
        m.stores = list(self.stores)
        m.arg_slots = dict(self.arg_slots)
        m.mem_slots = dict(self.mem_slots)
        return m

    # ------------------------------------------------------------------
    def relation(self, s: Store, seg, addr: T.Term, nbytes: int) -> str:
        """How does the store ``s`` relate to a ``nbytes`` access at ``addr``?"""
        if s.seg != seg:
            return DISJOINT
        b1, o1 = split(s.addr)
        b2, o2 = split(addr)
        if b1 is b2:
            s1, e1, s2, e2 = o1, o1 + s.size, o2, o2 + nbytes
            if e1 <= s2 or e2 <= s1:
                return DISJOINT
            if s1 <= s2 and e2 <= e1:
                return COVERS
            return OVERLAP
        st1, st2 = _is_stack(b1), _is_stack(b2)
        if st1 != st2:
            other_base = b2 if st1 else b1
            if other_base is None:                      # absolute address vs stack
                return DISJOINT
            off, size = (o1, s.size) if st1 else (o2, nbytes)
            if self.isolate_new_stack and off + size <= 0:
                return DISJOINT
        return UNKNOWN

    def load(self, seg, addr: T.Term, nbytes: int) -> T.Term:
        bits = nbytes * 8
        chain = self.stores
        for k in range(len(chain) - 1, -1, -1):
            s = chain[k]
            rel = self.relation(s, seg, addr, nbytes)
            if rel == DISJOINT:
                continue
            if rel == COVERS:
                _, so = split(s.addr)
                _, lo = split(addr)
                d = lo - so
                v = s.val
                if d == 0 and s.size == nbytes:
                    return v
                if d:
                    v = T.shr(v, T.const(8 * d, 8))
                return T.trunc(v, bits) if v.bits != bits else v
            # blocked: the value depends on memory contents we cannot forward
            s.pinned = True
            for j in range(k):
                if self.relation(chain[j], seg, addr, nbytes) != DISJOINT:
                    chain[j].pinned = True
            return T.load(seg, addr, s.seq, bits)
        if seg is None:                     # nothing was stored here - a declaration?
            declared = self.mem_at(addr, bits)
            if declared is not None:
                return declared
            arg = self.arg_at(addr, bits)
            if arg is not None:
                return arg
        return T.load(seg, addr, 0, bits)

    def arg_at(self, addr: T.Term, bits: int) -> T.Term | None:
        """The declared stack argument a read of ``bits`` bits at ``addr`` lands in,
        or None when it is not one.

        ``stack_offset`` measures against the *entry* ESP, so this survives pushes,
        pops and frame setup.  Only a read that fits inside a single slot counts - one
        that spans two of them stays an opaque load."""
        if not self.arg_slots:
            return None
        off = stack_offset(addr)
        if off is None:
            return None
        return self._slot(self.arg_slots, off, bits)

    def mem_at(self, addr: T.Term, bits: int) -> T.Term | None:
        """The declared memory value a read of ``bits`` bits at the absolute address
        ``addr`` lands in, or None (register based addresses never match)."""
        if not self.mem_slots:
            return None
        base, off = split(addr)
        if base is not None:
            return None
        return self._slot(self.mem_slots, off, bits)

    @staticmethod
    def _slot(slots: dict[int, T.Term], off: int, bits: int) -> T.Term | None:
        nbytes = bits // 8
        for base, slot in slots.items():
            if base <= off and off + nbytes <= base + DECL_SIZE:
                t = slot if off == base else T.shr(slot, T.const(8 * (off - base), 8))
                return t if t.bits == bits else T.trunc(t, bits)
        return None

    def may_read(self, seg, addr: T.Term, nbytes: int) -> tuple["Store", ...]:
        """The stores a load at ``addr`` can actually get its bytes from.

        This mirrors :meth:`load`: the newest non-disjoint store supplies the value
        when it covers the read; if the read is only partially covered (or the
        relation is unknown) every older store that may contribute is included.
        Stores that are already overwritten by a newer one are *not* dependencies -
        which is what makes dead-store elimination cascade.
        """
        chain = self.stores
        for k in range(len(chain) - 1, -1, -1):
            s = chain[k]
            rel = self.relation(s, seg, addr, nbytes)
            if rel == DISJOINT:
                continue
            if rel == COVERS:
                return (s,)
            return tuple(x for x in chain[:k + 1] if self.relation(x, seg, addr, nbytes) != DISJOINT)
        return ()

    def store(self, seg, addr: T.Term, nbytes: int, val: T.Term) -> Store:
        assert val.bits == nbytes * 8, (val, nbytes)
        s = Store(seg, addr, nbytes, val)
        self.stores.append(s)
        return s

    # ------------------------------------------------------------------
    def surviving_stores(self, final_esp_off):
        """Dead-store elimination for the stack.

        Returns ``(kept, late)``: ``kept`` are stores that must be emitted in
        program order; ``late`` are stores into the stub's own new stack area
        (entirely below the entry ESP, still above the final ESP) which can be
        emitted at the very end.  ``final_esp_off`` is the final ESP relative to
        the entry ESP, or ``None`` when ESP is not a plain offset (then nothing
        is eliminated).
        """
        stores = self.stores
        if final_esp_off is None:
            return list(stores), []
        needed = [False] * len(stores)
        covered: set[int] = set()
        # A byte is dead when the stack pointer ended above it *and* it is below
        # the entry ESP - only that part is the stub's own frame.  Memory at or
        # above the entry ESP belongs to the caller, however far ESP moved.
        dead_below = min(final_esp_off, 0)
        for idx in range(len(stores) - 1, -1, -1):
            s = stores[idx]
            off = stack_offset(s.addr) if s.seg is None else None
            if off is None:
                needed[idx] = True                      # non-stack stores always survive
                continue
            rng = range(off, off + s.size)
            live = any(b >= dead_below and b not in covered for b in rng)
            needed[idx] = live or s.pinned
            covered.update(rng)
        kept, late = [], []
        for s, n in zip(stores, needed):
            if not n:
                continue
            off = stack_offset(s.addr) if s.seg is None else None
            if off is not None and off + s.size <= 0 and not s.pinned:
                late.append(s)
            else:
                kept.append(s)
        return kept, late
