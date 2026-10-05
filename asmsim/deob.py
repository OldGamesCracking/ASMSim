"""Deobfuscating simplifier for x86-32 stubs.

The public entry point is :func:`simplify`::

    res = simplify(source, 0x01141AE7, regs_in=["ebx"], regs_out=["ebx"])
    open("handler1.bin", "wb").write(res.code)

``source`` is anything with ``read(addr, size) -> bytes`` - a snapshot file
(:class:`BytesSource`) or a live x64dbg session (:class:`RemoteSource`, driven by
x64dbg-automate).  Decoding follows the *real* control flow from ``start`` until
the ``is_end`` callback returns True (by default: an indirect ``jmp`` or a
``ret``); bytes that a ``jmp +1`` skips over therefore never become instructions,
which is exactly how the classic "jump over junk" obfuscation is removed.

Every instruction of that trace is then lifted with :mod:`asmsim.lift` and its
effect (registers, flags, memory) is derived from the resulting term DAG.  A
backward liveness pass keeps an instruction only if it

* writes one of the ``regs_out`` registers (or a register the terminating
  instruction reads, e.g. ``jmp eax`` keeps whatever computes eax),
* writes a flag some kept instruction reads,
* writes memory that survives (everything outside the stub's own new stack
  frame, plus stack slots some kept load reads), or
* is the terminating instruction itself.

Everything else - the junk that VM obfuscators use to hide the real data flow -
is dropped.  Runs of surviving instructions that only end up computing literals
are then collapsed into those literals (:func:`_fold_pass`), so an obfuscated
``mov eax, 1 / shl eax, 2`` comes out as ``mov eax, 4``.  The result is emitted
either

* ``inplace``: same start address, dropped bytes become NOPs, so the blob can be
  patched straight back into the original binary (internal addresses stay valid);
* ``compact``: only the surviving instructions, addresses shifted down, relative
  branches fixed up by iced's ``BlockEncoder``.

:func:`asmsim.verify.verify` differentially executes the original and the
simplified bytes against the Unicorn reference emulator.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Iterable, Mapping, Sequence

from iced_x86 import Decoder, FlowControl as FC, Instruction, InstructionInfoFactory
from iced_x86 import Mnemonic as MN
from iced_x86 import OpAccess, OpKind as OK, RflagsBits as RB

from . import terms as T
from .lift import FLAGS, REG32, REGINFO, Lifter, State, Unsupported
from .memory import DECL_SIZE, Store, stack_offset
from .x86asm import fmt as _fmt_ins

MAX_INSN = 15
_DECODE_WINDOW = 16

# transformation switches (mainly for tests / debugging)
FOLD = True
ESP_RUNS = True

# longest run of instructions a single fold may replace (bounds the pass's work
# and keeps whatever it rewrites local enough to still be readable)
MAX_FOLD_RUN = 24
# flags a ``xor reg, reg`` replacement would clobber
XOR_FLAGS = frozenset({"cf", "of", "zf", "sf", "pf", "af"})


# ===========================================================================================
# errors
# ===========================================================================================
class StubError(Exception):
    """Something is wrong with the stub itself (unreadable memory, bad decode)."""


class AnalysisStop(StubError):
    """The analysis could not continue; whatever was analysed up to here is kept."""

    def __init__(self, ip: int, why: str):
        self.ip, self.why = ip, why
        super().__init__(f"stopped at {ip:#x}: {why}")


class Loop(AnalysisStop):
    def __init__(self, ip, target):
        super().__init__(ip, f"control flow loops back to {target:#x} (not straight line)")


class NonLinear(AnalysisStop):
    def __init__(self, ip, cond=None):
        super().__init__(ip, f"conditional branch with non-constant condition ({cond})" if cond
                         else "conditional branch")


class CallOut(AnalysisStop):
    def __init__(self, ip, target):
        super().__init__(ip, f"call to {target:#x} leaves the stub (external call)")


class BadInstruction(AnalysisStop):
    def __init__(self, ip, why="invalid encoding"):
        super().__init__(ip, why)


# ===========================================================================================
# memory sources
# ===========================================================================================
class BytesSource:
    """A snapshot: ``data`` starts at ``base``."""

    def __init__(self, data: bytes, base: int):
        self.data, self.base = bytes(data), base

    def read(self, addr: int, size: int) -> bytes:
        off = addr - self.base
        if off < 0 or off + size > len(self.data):
            raise StubError(f"address {addr:#x}+{size:#x} outside snapshot "
                            f"{self.base:#x}..{self.base + len(self.data):#x}")
        return self.data[off:off + size]

    def close(self) -> None:
        pass


class RemoteSource:
    """View of a live process through an attached x64dbg-automate client.

    Reads are chunked at page boundaries; :meth:`write` patches memory - the
    caller decides when that is safe (a paused target, see :meth:`is_running`).
    """

    def __init__(self, client):
        self.client, self._closed = client, False

    def read(self, addr: int, size: int) -> bytes:
        out = bytearray()
        while size:
            n = min(size, 0x1000 - (addr & 0xFFF))
            chunk = bytes(self.client.read_memory(addr, n))
            if not chunk:
                raise StubError(f"cannot read {n} bytes at {addr:#x}")
            out += chunk
            addr += len(chunk)
            size -= len(chunk)
        return bytes(out)

    def write(self, addr: int, data: bytes) -> bool:
        """Patch ``len(data)`` bytes of the target's memory; False if any part failed."""
        data, off = bytes(data), 0
        while off < len(data):
            n = min(len(data) - off, 0x1000 - ((addr + off) & 0xFFF))
            if not self.client.write_memory(addr + off, data[off:off + n]):
                return False
            off += n
        return True

    def is_running(self) -> bool | None:
        """Whether the target is running - None when the client cannot say."""
        try:
            return bool(self.client.is_running())
        except Exception:
            return None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.client.detach_session()
            except Exception:
                pass


# ===========================================================================================
# the default terminator callback: indirect jmp or ret
# ===========================================================================================
def is_terminator(ins: Instruction) -> bool:
    """Default ``is_end``: the instruction is a ``jmp <reg>`` / ``jmp [mem]`` or a ``ret``."""
    if ins.mnemonic == MN.RET:
        return True
    return ins.mnemonic == MN.JMP and ins.op_kind(0) != OK.NEAR_BRANCH32


# ===========================================================================================
# trace records
# ===========================================================================================
@dataclasses.dataclass
class Load:
    seg: str | None
    addr: T.Term
    nbytes: int
    term: T.Term
    may_read: tuple[Store, ...]


@dataclasses.dataclass
class Step:
    ins: Instruction
    ip: int
    raw: bytes
    flow: object
    writes: frozenset[str]
    flag_writes: frozenset[str]
    reads: frozenset[str] = frozenset()
    flag_reads: frozenset[str] = frozenset()
    loads: tuple[Load, ...] = ()
    stores: tuple[Store, ...] = ()
    values: dict = dataclasses.field(default_factory=dict)   # written reg/flag -> new term
    pre: dict = dataclasses.field(default_factory=dict)      # written reg -> value before the step
    kind: str = ""                    # "" | "noop-jump" | "term"
    kept: bool = False
    reason: str = ""                  # why it was dropped
    rewritten: bool = False           # ins was folded to a simpler form
    orig: Instruction | None = None   # the instruction before folding
    repl: list[Instruction] | None = None   # replacement instructions (None: emit ins)
    fill_end: int | None = None       # end of the bytes a folded run owns (rest are NOPs)
    absorbed: bool = False            # folded into an earlier step's replacement
    esp_dropped: bool = False         # removed by the net-zero ESP pass
    esp_off: int | None = 0           # ESP after the step, relative to the entry ESP
    esp_off_before: int | None = 0    # ... and before it
    live_regs_after: frozenset = frozenset()
    live_flags_after: frozenset = frozenset()

    @property
    def end(self) -> int:
        return self.ip + len(self.raw)

    @property
    def region_end(self) -> int:
        """End of the bytes this step covers in the output.

        That is its own instruction unless it hosts a folded run, in which case it
        also owns the bytes of every instruction the run absorbed (NOPs)."""
        return self.fill_end if self.fill_end is not None else self.end

    def instrs(self) -> list[Instruction]:
        """The instructions this step emits (a folded run may emit several)."""
        return self.repl if self.repl is not None else [self.ins]

    def len_ops(self) -> int:
        return self.ins.op_count

    def encoded(self) -> bytes:
        """The bytes this step will emit (NOP padded up to its region)."""
        data = _encode_instrs(self.instrs(), self.ip)
        want = self.region_end - self.ip
        if len(data) < want and (self.rewritten or self.repl is not None):
            data += b"\x90" * (want - len(data))
        return data


@dataclasses.dataclass
class Trace:
    start: int
    steps: list[Step]
    state: State
    stop: AnalysisStop | None = None
    terminator: Step | None = None
    dead_bytes: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    # regions emitted verbatim: (start, end, the original bytes, the decoded
    # instructions).  Their instructions are still lifted and stepped - the trace
    # follows them normally - but nothing inside is rewritten or dropped.
    keep: list[tuple[int, int, bytes, list]] = dataclasses.field(default_factory=list)

    @property
    def end(self) -> int:
        return self.steps[-1].end if self.steps else self.start

    def in_keep(self, ip: int) -> bool:
        return any(lo <= ip < hi for lo, hi, _, _ in self.keep)


# ===========================================================================================
# term dependency walk (which registers / flags does a value depend on?)
# ===========================================================================================
def _term_deps(term: T.Term, memo: dict) -> tuple[frozenset, frozenset]:
    stack = [term]
    while stack:
        n = stack[-1]
        if n.uid in memo:
            stack.pop()
            continue
        pending = [a for a in n.operands() if a.uid not in memo]
        if pending:
            stack.extend(pending)
            continue
        regs: set[str] = set()
        flags: set[str] = set()
        if n.op == "in":
            # only real registers/flags: declared stack arguments are `in` terms too
            # (in:argN) but nothing produces them, and `regs_out` lookups must not see
            # them - e.g. when a terminator reads one (ret / jmp [esp])
            name = n.args[0]
            if n.bits == 1:
                if name in FLAGS:
                    flags.add(name)
            elif name in REG32:
                regs.add(name)
        for a in n.operands():
            r, f = memo[a.uid]
            regs |= r
            flags |= f
        memo[n.uid] = (frozenset(regs), frozenset(flags))
        stack.pop()
    return memo[term.uid]


_INFO = InstructionInfoFactory()
_FLAG_BITS = {"cf": RB.CF, "pf": RB.PF, "af": RB.AF, "zf": RB.ZF, "sf": RB.SF,
              "of": RB.OF, "df": RB.DF}


def arch_reads(ins: Instruction) -> tuple[frozenset, frozenset]:
    """Registers and flags the instruction *consumes at runtime*.

    Value dependencies alone are not enough: a register whose symbolic value
    happens to be a constant still has to hold that value when the instruction
    runs.  ``add [ecx], 4`` needs ecx to really contain the folded address, and
    ``cpuid`` reads eax even when the leaf number is a known constant - dropping
    its producer would send the real cpuid down a different path.
    """
    regs: set[str] = set()
    info = _INFO.info(ins)
    used = info.used_registers() if callable(info.used_registers) else info.used_registers
    for u in used:
        entry = REGINFO.get(u.register)
        if entry is None:
            continue
        name, bits, shift = entry
        if u.access in (OpAccess.NONE, OpAccess.WRITE):
            # a sub-register write (al, cl, bp, ...) merges into the full register,
            # so the instruction really reads the bits it does not overwrite
            if bits < 32 or shift:
                regs.add(name)
            continue
        regs.add(name)
    flags = frozenset(n for n, bit in _FLAG_BITS.items() if ins.rflags_read & bit)
    return frozenset(regs), flags


def term_deps(terms: Iterable[T.Term], memo: dict | None = None) -> tuple[frozenset, frozenset]:
    if memo is None:
        memo = {}
    regs: set[str] = set()
    flags: set[str] = set()
    for t in terms:
        r, f = _term_deps(t, memo)
        regs |= r
        flags |= f
    return frozenset(regs), frozenset(flags)


# ===========================================================================================
# discovery + symbolic lifting (one pass - the flow decides where the next instruction is)
# ===========================================================================================
def analyse(src, start: int, is_end: Callable[[Instruction], bool] = is_terminator, *,
            concrete: Mapping[str, int] | None = None, args: Mapping[int, T.Term] | None = None,
            mem: Mapping[int, T.Term] | None = None,
            max_instructions: int = 20000) -> Trace:
    """Decode and lift the stub from ``start`` until ``is_end`` says stop."""
    st = State.initial(concrete=dict(concrete or {}), args=dict(args or {}),
                       mem=dict(mem or {}))
    lifter = Lifter(st)
    steps: list[Step] = []
    dead: list[tuple[int, int]] = []
    seen: set[int] = set()
    memo: dict = {}
    ip, stop, term = start, None, None

    while len(steps) < max_instructions:
        if ip in seen:
            stop = Loop(ip, ip)
            break
        seen.add(ip)
        try:
            window = src.read(ip, _DECODE_WINDOW)
        except StubError as e:
            stop = AnalysisStop(ip, str(e))
            break
        ins = Decoder(32, window, ip=ip).decode()
        if ins.len <= 0 or ins.is_invalid:
            stop = BadInstruction(ip, f"invalid encoding {window[:8].hex()}")
            break
        raw = window[:ins.len]
        if steps and steps[-1].end < ip:
            dead.append((steps[-1].end, ip))

        pre_regs, pre_flags = dict(st.regs), dict(st.flags)
        loads: list[Load] = []
        orig_load = st.mem.load

        def recording_load(seg, addr, nbytes, _orig=orig_load, _loads=loads, _st=st):
            may = _st.mem.may_read(seg, addr, nbytes)
            t = _orig(seg, addr, nbytes)
            _loads.append(Load(seg, addr, nbytes, t, may))
            return t

        n_stores = len(st.mem.stores)
        st.mem.load = recording_load
        try:
            flow = lifter.step(ins)
        except Unsupported as e:
            stop = AnalysisStop(ip, f"unsupported instruction: {e.ins} ({e.why or 'no semantics'})")
            break
        finally:
            del st.mem.load

        writes = frozenset(n for n in REG32 if st.regs[n] is not pre_regs[n])
        fwrites = frozenset(n for n in FLAGS if st.flags[n] is not pre_flags[n])
        new_stores = tuple(st.mem.stores[n_stores:])

        roots = [st.regs[n] for n in writes] + [st.flags[n] for n in fwrites]
        for s in new_stores:
            roots += [s.addr, s.val]
        roots += [l.addr for l in loads]
        reads, flag_reads = term_deps(roots, memo)
        areads, aflags = arch_reads(ins)
        reads |= areads
        flag_reads |= aflags

        step = Step(ins=ins, ip=ip, raw=raw, flow=flow, writes=writes, flag_writes=fwrites,
                    reads=reads, flag_reads=flag_reads, loads=tuple(loads), stores=new_stores,
                    values={n: st.regs[n] for n in writes} | {f: st.flags[f] for f in fwrites},
                    pre={n: pre_regs[n] for n in writes},
                    esp_off=stack_offset(st.regs["esp"]),
                    esp_off_before=stack_offset(pre_regs["esp"]))
        steps.append(step)

        if is_end(ins):
            step.kind, term = "term", step
            break
        if flow.kind == "next":
            ip = ins.next_ip
        elif flow.kind == "jump":
            if flow.target <= ip:
                stop = Loop(ip, flow.target)
                break
            step.kind = "noop-jump"          # forward jump over junk - execution continues at target
            ip = flow.target
        elif flow.kind in ("indirect", "halt"):
            step.kind, term = "term", step
            break
        elif flow.kind == "branch":
            stop = NonLinear(ip, T.fmt(flow.cond, 3))
            break
        else:  # pragma: no cover - defensive
            stop = AnalysisStop(ip, f"unhandled flow {flow.kind}")
            break

    trace = Trace(start=start, steps=steps, state=st, stop=stop, terminator=term, dead_bytes=dead)
    _terminator_reads(trace, memo)
    return trace


def _terminator_reads(trace: Trace, memo: dict) -> None:
    """Registers/flags the final instruction reads.

    Memory operands (``ret`` pops, ``jmp [mem]``) were already recorded by the
    load hook while the instruction was lifted, so only register operands and the
    indirect flow's term are left.
    """
    term = trace.terminator
    if term is None:
        return
    regs: set[str] = set()
    flags: set[str] = set()
    ins = term.ins
    for k in range(ins.op_count):
        kind = ins.op_kind(k)
        if kind == OK.REGISTER:
            info = REGINFO.get(ins.op_register(k))
            if info and info[2] == 0:                 # 32/16/8-low view -> the 32 bit register
                regs.add(info[0])
        elif kind == OK.MEMORY:
            for r in (ins.memory_base, ins.memory_index):
                info = REGINFO.get(r)
                if info:
                    regs.add(info[0])
    flow_term = getattr(term.flow, "term", None)
    if isinstance(flow_term, T.Term):
        r, f = _term_deps(flow_term, memo)
        regs |= r
        flags |= f
    areads, aflags = arch_reads(ins)
    term.reads = frozenset(term.reads | regs | areads)
    term.flag_reads = frozenset(term.flag_reads | flags | aflags)


# ===========================================================================================
# slicing
# ===========================================================================================
def _load_result_live(step: Step) -> bool:
    """Does the value a kept load produces reach a live register?

    A ``pop esi`` whose forwarded value happens to equal ESI's current value writes
    nothing *new* symbolically (its ``writes`` is just ``esp``), but it still writes
    ESI at runtime - so the store it re-reads has to stay in the output.  A memory
    destination (``pop [esp+4]``, ``add [esp], ebx``) needs no case here: there the
    loaded value flows into the step's own store, which the step's store liveness in
    :func:`slice_trace` already covers.
    """
    if not step.loads:
        return False
    ins = step.ins
    if ins.op_count and ins.op_kind(0) == OK.REGISTER:
        info = REGINFO.get(ins.op_register(0))
        if info and info[0] in step.live_regs_after:
            return True
    return False


def slice_trace(trace: Trace, regs_out: Sequence[str], keep_unread_stores: bool = True,
                protect_loaded_stores: bool = True) -> dict:
    """Backward liveness; marks ``step.kept``.  Returns statistics.

    ``protect_loaded_stores`` keeps a store whose bytes a *surviving* instruction
    still reads at runtime (see :func:`_load_result_live`).  Turning it off restores
    the older, smaller - but unsound - dead-store elimination: the emitted load then
    re-reads memory the dropped store was supposed to have written.
    """
    steps, st = trace.steps, trace.state
    complete = trace.terminator is not None

    stats = {"stores_kept": 0, "stores_dropped": 0}
    if not complete:
        # the tail was not analysed - everything at the stop point may be observable
        live_regs = set(REG32)
        live_flags = set(FLAGS)
        live_stores = {id(s) for s in st.mem.stores}
    else:
        final_off = stack_offset(st.regs["esp"]) if st.regs["esp"].op != "undef" else None
        kept_stores, late_stores = st.mem.surviving_stores(final_off)
        live_stores = {id(s) for s in kept_stores + late_stores}
        live_stores |= {id(s) for s in st.mem.stores if s.pinned}
        if not keep_unread_stores:
            # aggressive: only stores that some load may read are observable
            live_stores = {id(s) for s in st.mem.stores if s.pinned}
        live_regs = set(regs_out) | set(trace.terminator.reads)
        live_flags = set(trace.terminator.flag_reads)

    # stores the memory model says must survive (observable bytes, or a load that
    # could not be forwarded) - the backward walk below only adds more
    required_stores = set(live_stores)
    for step in reversed(steps):
        step.live_regs_after = frozenset(live_regs)
        step.live_flags_after = frozenset(live_flags)
        if step.kind == "term":
            keep = True
        elif step.absorbed:
            keep = False                   # its effect is already in an earlier step's rewrite
        else:
            keep = (bool(step.writes & live_regs) or bool(step.flag_writes & live_flags)
                    or any(id(s) in live_stores for s in step.stores))
        step.kept = keep
        if keep:
            step.reason = ""
            step.esp_dropped = False
        if keep:
            if step.kind == "term":
                value_live = True
            else:
                # standard kill: the instruction overwrites its destinations, so their
                # previous value is not needed before it (the value *after* it was)
                live_regs = (live_regs - step.writes) | step.reads
                live_flags = (live_flags - step.flag_writes) | step.flag_reads
                value_live = (bool((step.writes - {"esp"}) & step.live_regs_after)
                              or bool(step.flag_writes & step.live_flags_after)
                              or any(id(s) in live_stores for s in step.stores)
                              or ("esp" in step.live_regs_after and step.writes and step.esp_off is None))
            if value_live or (protect_loaded_stores and _load_result_live(step)):
                # the step is emitted, so its loads happen at runtime: whatever they
                # forward from has to be there
                for load in step.loads:
                    for s in load.may_read:
                        live_stores.add(id(s))
        if not keep:
            if not step.esp_dropped and not step.absorbed:
                bits = []
                if step.writes:
                    bits.append("writes " + ",".join(sorted(step.writes)))
                if step.flag_writes:
                    bits.append("flags " + ",".join(sorted(step.flag_writes)))
                if step.stores:
                    bits.append(f"{len(step.stores)} store(s)")
                if step.kind == "noop-jump":
                    bits.append("jump to next instruction")
                step.reason = "dead (" + ("; ".join(bits) or "no effect") + ")"
            continue
    stats["stores_kept"] = sum(1 for s in st.mem.stores if id(s) in live_stores)
    stats["stores_dropped"] = len(st.mem.stores) - stats["stores_kept"]
    stats["ins_total"] = len(steps)
    stats["ins_kept"] = sum(1 for s in steps if s.kept)
    stats["ins_dropped"] = stats["ins_total"] - stats["ins_kept"]
    stats["dead_bytes"] = sum(e - b for b, e in trace.dead_bytes)
    stats["_live_stores"] = live_stores
    stats["_required_stores"] = required_stores
    return stats


def _materialize(reg: str, val: T.Term, entry: T.Term | None, xor_ok: bool) -> Instruction | None:
    """One instruction that turns ``entry`` (the register before the run) into
    ``val`` (its value after it), or None when no single instruction can.

    Two shapes are handled, both of which write state derived from the run's
    *entry* state only:

    * a constant - ``mov reg, imm32`` (or the 2 byte ``xor reg, reg`` for zero),
    * a partial write - ``mov cl, imm8`` / ``mov ch, imm8`` / ``mov cx, imm16``
      keep the rest of the register, so it is emitted when writing the literal
      over ``entry`` yields exactly the run's value.  The entry byte may be
      anything (a loaded byte, a previous blend), it only has to agree with the
      result *outside* the written field.
    """
    if val.op == "c":
        v = val.args[0]
        if v == 0 and xor_ok:
            return xor_zero(reg)                       # 2 bytes instead of 5
        return mov_const(reg, v)
    if val.op == "ins" and entry is not None:
        part, shift = val.args[1], val.args[2]
        if part.op != "c":
            return None
        insn = subreg_mov(reg, part, shift)
        if insn is not None and T.ins(entry, part, shift) is val:
            return insn
    return None


def _fold_pass(trace: Trace, stats: dict, max_run: int = MAX_FOLD_RUN) -> int:
    """Collapse runs of kept instructions into direct register writes.

    This is what removes the ``push imm32 / pop reg / add reg, imm32`` idiom the VM
    uses to write constants, and the ``mov eax, 1 / shl eax, 2`` chains on top of
    it: whenever the value a register ends a run with is a literal (or a literal
    spliced into its own entry value), the whole run is replaced by that write.

    The run's first kept instruction hosts the replacement instructions (padding
    the rest of the run with NOPs), the others are absorbed.  Safe because

    * the replacement depends only on pre-run state - a constant, or a partial
      write that merges into the register's entry value,
    * no flag the run writes is read after it (so dropping the flag updates is
      invisible, and ``xor reg, reg`` may only be used when its flags are dead),
    * every store it drops is either one the slicer already found dead, or one whose
      only readers are inside the run - so store and load disappear together.

    Iterated together with the slice until nothing changes.
    """
    steps = trace.steps
    live_stores = stats["_live_stores"]
    required = stats["_required_stores"]
    # which steps may read each store - a run may only drop a store when every one
    # of its readers goes with the run (see _Run.plan)
    readers: dict[int, tuple[int, int]] = {}
    for k, s in enumerate(steps):
        for load in s.loads:
            for src in load.may_read:
                lo, hi = readers.get(id(src), (k, k))
                readers[id(src)] = (min(lo, k), max(hi, k))
    count = 0
    i, n = 0, len(steps)
    while i < n:
        s = steps[i]
        if not s.kept or s.absorbed or s.kind or s.repl is not None:
            i += 1
            continue
        if trace.in_keep(s.ip):               # a verbatim region is nobody's to rewrite
            i += 1
            continue
        win = _Run(steps, i, live_stores, required, readers)
        best = None
        run = 0
        j = i
        while j < n:
            s = steps[j]
            if s.kind or trace.in_keep(s.ip):  # a jump/terminator/kept region: never across
                break
            if s.absorbed or not s.kept:      # dropped bytes are NOPs, a run may span them
                j += 1
                continue
            if s.repl is not None or run >= max_run:
                break
            if not win.add(j):
                break
            run += 1
            plan = win.plan(j)
            if plan is not None:
                best = (j, plan)
            j += 1
        if best is None or not _install(trace, i, best[0], best[1]):
            i += 1
            continue
        count += 1
        i = best[0] + 1
    return count


class _Run:
    """Incremental bookkeeping for a candidate run of consecutive kept steps."""

    __slots__ = ("steps", "start", "live_stores", "required", "readers",
                 "final", "entry", "flags", "moves")

    def __init__(self, steps: list[Step], start: int, live_stores: set,
                 required: set, readers: dict):
        self.steps, self.start = steps, start
        self.live_stores, self.required, self.readers = live_stores, required, readers
        self.final: dict[str, T.Term] = {}      # register -> its value at the run's end
        self.entry: dict[str, T.Term | None] = {}   # ... and before it
        self.flags: set[str] = set()
        self.moves: list[Store] = []            # stores the run would drop

    def add(self, j: int) -> bool:
        s = self.steps[j]
        for x in s.stores:
            if id(x) in self.required:
                return False                    # observable (or blocking) - the run ends here
            self.moves.append(x)                # droppable if its readers go with the run
        for r in s.writes:
            self.final[r] = s.values[r]
            if r not in self.entry:
                self.entry[r] = s.pre.get(r)
        self.flags |= s.flag_writes
        return True

    def plan(self, j: int) -> list[tuple[str, T.Term, Instruction, frozenset]] | None:
        """What to emit so the run ending at step ``j`` has the same visible effect."""
        after = self.steps[j]
        live, live_flags = after.live_regs_after, after.live_flags_after
        if self.flags & live_flags:
            return None                         # a flag the run writes is still needed
        for x in self.moves:                    # a store the run would drop ...
            if id(x) in self.live_stores:       # ... that something still reads
                lo, hi = self.readers.get(id(x), (0, 0))
                if lo < self.start or hi > j:
                    return None                 # that reader outlives the run
        xor_ok = not (XOR_FLAGS & live_flags)   # ... including the ones "xor reg,reg" would
        out = []
        for reg, val in self.final.items():
            if reg not in live:
                continue                        # no later instruction wants this register
            entry = self.entry.get(reg)
            if val is entry:
                continue                        # the run leaves the register as it was
            ins = _materialize(reg, val, entry, xor_ok)
            if ins is None:
                return None
            flags = XOR_FLAGS if ins.mnemonic == MN.XOR else frozenset()
            out.append((reg, val, ins, flags))
        ip = self.steps[self.start].ip
        if len(_encode_instrs([p[2] for p in out], ip)) > self.steps[j].end - ip:
            return None                         # the replacement would not fit in the run
        return out


def _emitted_reads(insn: Instruction) -> frozenset[str]:
    """Registers an emitted replacement instruction really reads.

    A partial write (``mov bl, 0``) merges into the bits it does not overwrite,
    so the register's own producer has to survive; a ``xor reg, reg`` reads
    nothing at all, so it must not keep a dead producer alive.
    """
    if (insn.mnemonic == MN.XOR and insn.op_count == 2 and insn.op_kind(0) == OK.REGISTER
            and insn.op_kind(1) == OK.REGISTER and insn.op_register(0) == insn.op_register(1)):
        return frozenset()
    regs, flags = arch_reads(insn)
    return frozenset(regs | flags)


def _install(trace: Trace, i: int, j: int, plan: list) -> bool:
    """Apply a fold of the run ``steps[i..j]``, hosted by ``steps[i]``."""
    steps = trace.steps
    host = steps[i]
    insns = [p[2] for p in plan]
    if insns and j == i and _encode_instrs(insns, host.ip) == _encode_instr(host.ins, host.ip):
        return False                            # same bytes: not worth a rewrite
    note = (f"folded into the run at {host.ip:#x}" if insns
            else "dead run (nothing outside it reads its result)")
    if insns:
        host.orig = host.ins
        host.repl = insns
        host.ins = insns[0]
        host.fill_end = steps[j].end
        host.rewritten = True
        host.writes = frozenset(p[0] for p in plan)
        host.values = {p[0]: p[1] for p in plan}
        host.flag_writes = frozenset().union(*(p[3] for p in plan))
        host.flag_reads = frozenset()
        # what the replacement reads: the dependencies of its values, plus - for a
        # partial write - the register's own bits it merges into, whose producer
        # must therefore stay in the output
        reads, _ = term_deps([p[1] for p in plan])
        host.reads = frozenset(reads) | frozenset().union(
            *(_emitted_reads(p[2]) for p in plan))
        host.loads = ()
        host.stores = ()
    # Every step inside the run's bytes must be absorbed, including ones the
    # slicer had dropped: the host owns their bytes now, and a later slice must
    # not be able to keep one of them on top of the replacement.
    for k in range(i, j + 1):
        if k == i and insns:
            continue
        s = steps[k]
        if s.absorbed:
            continue
        s.kept = False                          # NOPs in place, gone in compact mode
        s.absorbed = True
        s.reason = note
        s.writes = s.flag_writes = s.reads = s.flag_reads = frozenset()
        s.stores = s.loads = ()
    return True


def _drop_esp_runs(trace: Trace, stats: dict) -> int:
    """Drop stacks of instructions whose only effect was moving ESP by zero.

    The VM writes constants as ``push imm32 / pop reg`` and saves registers with
    ``push reg / ... / pop reg``; once the payload is dead those runs touch
    nothing but a dead stack slot, and because their net ESP change is zero (and
    no surviving store/load lives inside) removing them is invisible.
    """
    live_stores = stats["_live_stores"]
    kept = [s for s in trace.steps if s.kept]
    removed = 0
    run: list[Step] = []

    def flush():
        nonlocal removed
        if run and run[0].esp_off_before is not None and run[-1].esp_off is not None \
                and run[0].esp_off_before == run[-1].esp_off:
            for s in run:
                s.kept = False
                s.esp_dropped = True
                s.reason = "stack pointer dance (net ESP change 0, nothing live)"
            removed += len(run)
        run.clear()

    for s in kept:
        essential = (bool((s.writes - {"esp"}) & s.live_regs_after)
                     or bool(s.flag_writes & s.live_flags_after)
                     or any(id(x) in live_stores for x in s.stores)
                     or s.kind == "term")
        if essential or s.esp_off is None or s.esp_off_before is None:
            flush()
        else:
            run.append(s)
    flush()
    return removed


def mov_const(reg: str, value: int) -> Instruction:
    from iced_x86 import Code, Instruction as I, Register
    return I.create_reg_i32(Code.MOV_R32_IMM32, getattr(Register, reg.upper()),
                            T.sx(value & 0xFFFFFFFF, 32))     # iced wants a signed imm32


def xor_zero(reg: str) -> Instruction:
    from iced_x86 import Code, Instruction as I, Register
    return I.create_reg_reg(Code.XOR_RM32_R32, getattr(Register, reg.upper()),
                            getattr(Register, reg.upper()))


_BYTE_REGS = ("eax", "ecx", "edx", "ebx")


def subreg_mov(reg: str, part: T.Term, shift: int) -> Instruction | None:
    """``mov`` into the register view that covers ``part.bits`` bits at ``shift``.

    ``ins(entry, const, 0/8)`` describes a write to al/cl/dl/bl or ah/ch/dh/bh,
    ``ins(entry, const, 0)`` with 16 bits a write to ax/cx/... - None when the
    register has no view of that shape (there is no 8 bit register for esp, ...).
    """
    from iced_x86 import Code, Instruction as I, Register
    v = part.args[0]
    if part.bits == 16 and shift == 0:
        return I.create_reg_i32(Code.MOV_R16_IMM16, getattr(Register, reg[1:].upper()), v)
    if part.bits == 8 and reg in _BYTE_REGS:
        name = reg[1] + ("h" if shift == 8 else "l")        # eax -> al / ah
        return I.create_reg_i32(Code.MOV_R8_IMM8, getattr(Register, name.upper()), v)
    return None


# ===========================================================================================
# emission
# ===========================================================================================
def _encode_instrs(instrs: Iterable[Instruction], ip: int) -> bytes:
    from iced_x86 import BlockEncoder
    enc = BlockEncoder(32)
    for i in instrs:
        enc.add(i)
    return bytes(enc.encode(ip))


def _encode_instr(ins: Instruction, ip: int) -> bytes:
    return _encode_instrs((ins,), ip)


def emit_inplace(trace: Trace) -> bytes:
    """Same addresses; dropped instructions/bytes become NOPs, kept regions verbatim."""
    start, end = trace.start, trace.end
    buf = bytearray(b"\x90" * (end - start))
    for step in trace.steps:
        if not step.kept or trace.in_keep(step.ip):
            continue
        if any(step.region_end > lo and step.ip < hi for lo, hi, _, _ in trace.keep):
            raise StubError(f"the folded run at {step.ip:#x} reaches into a kept region")
        data = step.encoded()
        if len(data) != step.region_end - step.ip:
            raise StubError(f"instruction at {step.ip:#x} re-encodes to {len(data)} bytes, "
                            f"expected {step.region_end - step.ip} (use compact mode)")
        buf[step.ip - start:step.region_end - start] = data
    for lo, hi, data, _ in trace.keep:          # the original bytes, last word
        buf[lo - start:hi - start] = data
    return bytes(buf)


def compact_instrs(trace: Trace, upto: Step | None = None) -> list:
    """The instruction list a compact blob is built from: the surviving steps plus the
    kept regions, in address order (everything before ``upto`` when given)."""
    out: list[Instruction] = []
    pending = list(trace.keep)
    for step in trace.steps:
        while pending and step.ip >= pending[0][1]:     # the region comes before this step
            out += pending.pop(0)[3]
        if step is upto:
            break
        if step.kept and not trace.in_keep(step.ip):
            out += step.instrs()
    if upto is None:
        for _, _, _, instrs in pending:
            out += instrs
    return out


def emit_compact(trace: Trace) -> bytes:
    """Only the surviving instructions, starting at the original start address."""
    if trace.terminator is None:
        raise StubError("analysis stopped early - compact output would invalidate addresses; "
                        "use inplace mode")
    return bytes(_encode_instrs(compact_instrs(trace), trace.start))


# ===========================================================================================
# the result
# ===========================================================================================
@dataclasses.dataclass
class Result:
    start: int
    end: int
    mode: str
    code: bytes
    trace: Trace
    stats: dict
    outputs: dict[str, T.Term]
    compare_regs: tuple[str, ...] = ()
    concrete: dict[str, int] = dataclasses.field(default_factory=dict)
    concrete_args: dict[int, int] = dataclasses.field(default_factory=dict)   # offset -> value
    concrete_mem: dict[int, int] = dataclasses.field(default_factory=dict)    # address -> value

    @property
    def kept_steps(self) -> list[Step]:
        return [s for s in self.trace.steps if s.kept]

    @property
    def terminator_ip_out(self) -> int | None:
        """Address of the terminating instruction in the emitted blob."""
        from iced_x86 import BlockEncoder
        term = self.trace.terminator
        if term is None:
            return None
        if self.mode == "inplace":
            return term.ip                       # dropped bytes become NOPs, addresses stay
        prefix = compact_instrs(self.trace, upto=term)
        return self.start + len(_encode_instrs(prefix, self.start))

    @property
    def instructions(self) -> list[Instruction]:
        return [i for s in self.kept_steps for i in s.instrs()]

    def patch_bytes(self) -> bytes:
        """The bytes that replace the original stub at :attr:`start`.

        ``inplace`` output already spans ``start..end`` (whatever was dropped is a
        NOP); a ``compact`` blob is padded the same way, so patching leaves no
        stale instruction of the original stub behind.
        """
        return self.code + b"\x90" * (self.end - self.start - len(self.code))

    def listing(self, show_dropped: bool = True, annotate: int = 1) -> str:
        """Readable listing of the simplified stub."""
        if annotate:
            memo: dict = {}
        lines = []
        dropped_run = 0
        # every emitted instruction gets its own line, so size the column for the
        # longest single one
        width = max([26] + [3 * len(_encode_instr(i, s.ip)) - 1
                            for s in self.trace.steps if s.kept for i in s.instrs()]
                    + [3 * len(s.raw) - 1 for s in self.trace.steps])
        pending_keep = list(self.trace.keep)

        def flush_keep(before: int) -> None:
            """Print the kept regions that start at or before ``before``."""
            nonlocal dropped_run
            while pending_keep and before >= pending_keep[0][0]:
                lo, hi, _data, instrs = pending_keep.pop(0)
                if dropped_run:
                    lines.append(f"; --- {dropped_run} instruction(s) removed ---")
                    dropped_run = 0
                lines.append(f"; --- kept verbatim {lo:08X}..{hi:08X} ---")
                ip = lo
                for reg in instrs:
                    data = _encode_instr(reg, ip)
                    lines.append(f"{ip:08X} {' '.join(f'{b:02x}' for b in data):<{width}} "
                                 f"{_fmt_ins(reg)}")
                    ip += len(data)
                lines.append("; --- end of kept region ---")

        for step in self.trace.steps:
            flush_keep(step.ip)
            if self.trace.in_keep(step.ip):
                continue                        # shown as part of the region above
            if not step.kept:
                dropped_run += 1
                if show_dropped:
                    lines.append(f"{step.ip:08X} {' '.join(f'{b:02x}' for b in step.raw):<{width}} "
                                 f"; {step.reason}: {_fmt_ins(step.ins)}")
                continue
            if dropped_run:
                lines.append(f"; --- {dropped_run} instruction(s) removed ---")
                dropped_run = 0
            note = ""
            if annotate:
                vals = [f"{n}={T.fmt(t, annotate)}" for n, t in sorted(step.values.items())
                        if annotate >= 3 or t.bits != 1]
                if step.kind == "term":
                    vals.append("EXIT")
                note = "  ; " + ", ".join(vals) if vals else ""
            if step.rewritten:
                note += f"   [folded from: {_fmt_ins(step.orig)}]"
            # a folded run can emit several instructions - one line each, at the
            # address it will actually run from
            ip = step.ip
            for n, ins in enumerate(step.instrs()):
                data = _encode_instr(ins, ip)
                lines.append(f"{ip:08X} {' '.join(f'{b:02x}' for b in data):<{width}} "
                             f"{_fmt_ins(ins)}{note if n == 0 else ''}")
                ip += len(data)
        flush_keep(1 << 64)
        if dropped_run:
            lines.append(f"; --- {dropped_run} instruction(s) removed ---")
        return "\n".join(lines)

    def summary(self) -> str:
        s = self.stats
        head = (f"stub {self.start:#x}..{self.end:#x}  "
                f"{s['ins_total']} instructions -> {s['ins_kept']} kept, {s['ins_dropped']} removed "
                f"({self.end - self.start} -> {len(self.code)} bytes, mode={self.mode})")
        lines = [head]
        if s.get("folded"):
            lines.append(f"  folded to literals: {s['folded']}"
                         + (f" ({s['absorbed']} instruction(s) absorbed)" if s.get("absorbed") else ""))
        if s.get("args"):
            lines.append(f"  stack args: {s['args']}")
        if s.get("mem"):
            lines.append(f"  known memory: {s['mem']}")
        if s.get("keep"):
            lines.append(f"  kept verbatim: {s['keep']}")
        if s.get("esp_runs"):
            lines.append(f"  removed as dead stack-pointer dances: {s['esp_runs']}")
        if s["dead_bytes"]:
            lines.append(f"  dead bytes skipped by jumps: {s['dead_bytes']}")
        lines.append(f"  memory stores: {s['stores_kept']} kept, {s['stores_dropped']} removed")
        if not s.get("protect_loaded_stores", True):
            lines.append("  !! --unsafe-stores: a surviving load may read a store that "
                         "was dropped")
        for name, term in self.outputs.items():
            lines.append(f"  {name} = {T.fmt(term, 8)}")
        if self.trace.stop is not None:
            lines.append(f"  !! analysis stopped early: {self.trace.stop}")
        return "\n".join(lines)


# ===========================================================================================
# top level
# ===========================================================================================
def _norm_regs(regs) -> dict[str, int | None]:
    if regs is None:
        return {}
    if isinstance(regs, Mapping):
        return {str(k).lower(): v for k, v in regs.items()}
    return {str(r).lower(): None for r in regs}


def _norm_keep(keep, start: int) -> list[tuple[int, int]]:
    """``[(lo, hi), ...]`` (or ``"0x...-0x..."`` strings) -> validated half open ranges."""
    out: list[tuple[int, int]] = []
    if keep is None:
        return out
    items = [p for p in (x.strip() for x in keep.split(",")) if p] if isinstance(keep, str) \
        else list(keep)
    for item in items:
        if isinstance(item, str):
            lo_s, _, hi_s = item.partition("-")
            if not hi_s:
                raise ValueError(f"keep region {item!r} needs LO-HI")
            lo, hi = int(lo_s, 0), int(hi_s, 0)
        else:
            lo, hi = item
        if lo < start or hi <= lo:
            raise ValueError(f"keep region {lo:#x}-{hi:#x} is not inside the stub "
                             f"(starts at {start:#x})")
        out.append((lo, hi))
    ordered = sorted(out)
    for (_, hi), (lo2, _) in zip(ordered, ordered[1:]):
        if lo2 < hi:
            raise ValueError(f"keep regions {hi:#x} and {lo2:#x} overlap")
    return ordered


def _read_keep(source, ranges: list[tuple[int, int]], end: int, mode: str) -> list:
    """Fetch the bytes of each kept region and decode them: the bytes are what an
    inplace blob gets verbatim, the instructions what a compact blob re-encodes.

    A compact blob moves code, so every branch inside a region has to stay inside it
    (its own displacement survives); one that leaves the region cannot be re-targeted
    and is refused instead of being emitted wrong."""
    out = []
    for lo, hi in ranges:
        if lo >= end:
            raise ValueError(f"keep region {lo:#x}-{hi:#x} starts after the stub ends")
        hi = min(hi, end)
        data = source.read(lo, hi - lo)
        instrs, ip = [], lo
        for ins in Decoder(32, data, ip=lo):
            if ins.is_invalid or ip + ins.len > hi:
                break
            if mode == "compact" and ins.flow_control in (FC.UNCONDITIONAL_BRANCH,
                                                          FC.CONDITIONAL_BRANCH, FC.CALL):
                target = ins.near_branch_target
                if not lo <= target <= hi:
                    raise ValueError(
                        f"keep region {lo:#x}-{hi:#x} branches out of itself "
                        f"({_fmt_ins(ins)} -> {target:#x}); a compact blob would move it "
                        f"- use inplace mode")
            instrs.append(ins)
            ip += ins.len
        if not instrs or ip != hi:
            raise ValueError(f"keep region {lo:#x}-{hi:#x} does not end on an instruction "
                             f"boundary (the last one ends at {ip:#x})")
        out.append((lo, hi, bytes(data), instrs))
    return out


def _norm_mem(mem) -> dict[int, T.Term]:
    """Known memory contents as ``{absolute address: value}`` (a dword each).

    Same shapes as :func:`_norm_regs`: a list of addresses, or a mapping address ->
    value (``None`` meaning "known to be there, but not which value").  These resolve
    data dependent branches - e.g. the spin lock byte a stub waits on."""
    slots: dict[int, T.Term] = {}
    if mem is None:
        return slots
    items = mem.items() if isinstance(mem, Mapping) else ((a, None) for a in mem)
    for addr, val in items:
        a = (int(addr, 0) if isinstance(addr, str) else int(addr)) & 0xFFFFFFFF
        if a in slots:
            raise ValueError(f"memory at {a:#x} is declared twice")
        slots[a] = T.const(val, 32) if val is not None else T.inp(f"mem{a:#x}", 32)
    # a declaration covers one dword, so two of them must not overlap
    ordered = sorted(slots)
    for a, b in zip(ordered, ordered[1:]):
        if b < a + DECL_SIZE:
            raise ValueError(f"declared memory at {a:#x} and {b:#x} overlap")
    return slots


def _norm_args(args) -> dict[int, T.Term]:
    """``arg0`` -> ``[entry_esp+0]``, ``arg1`` -> ``[entry_esp+4]``, ...

    Takes what :func:`_norm_regs` takes (a list of names, or a mapping name ->
    concrete value) and returns the slot offset -> the term the slot holds: an input
    for a symbolic argument, a constant for a known one."""
    slots: dict[int, T.Term] = {}
    if args is None:
        return slots
    items = args.items() if isinstance(args, Mapping) else ((a, None) for a in args)
    for name, val in items:
        n = str(name).lower()
        if not (n.startswith("arg") and n[3:].isdigit()):
            raise ValueError(f"unknown stack argument {name!r} (expected arg0, arg1, ...)")
        off = DECL_SIZE * int(n[3:])
        if off in slots:
            raise ValueError(f"stack argument {name!r} is declared twice")
        arg = f"arg{off // DECL_SIZE}"                   # canonical, so arg007 == arg7
        slots[off] = T.const(val, 32) if val is not None else T.inp(arg, 32)
    return slots


def simplify(source, start: int, regs_in=(), regs_out=(), is_end: Callable | None = None, *,
             concrete: Mapping[str, int] | None = None, mode: str = "inplace",
             max_instructions: int = 20000, keep_unread_stores: bool = True,
             protect_loaded_stores: bool = True, args=None, mem=None, keep=None) -> Result:
    """Lift, simplify and re-emit the stub at ``start``.

    ``regs_in``   registers whose entry value is an input (a list of names, or a
                  mapping name -> concrete value; ``None`` means symbolic).
    ``regs_out``  registers whose final value must be preserved.
    ``args``      the stub's stack arguments, ``arg0`` at ``[esp+0]``, ``arg1`` at
                  ``[esp+4]``, ... - the same shapes as ``regs_in``.  Reads of those
                  slots become ``in:argN`` terms (or constants for known values,
                  which the fold pass can then materialise).
    ``mem``       known memory contents, ``{absolute address: value}`` (a dword each,
                  like ``args``).  A known value resolves data dependent branches -
                  e.g. the lock byte a stub spins on - so the trace can continue.
    ``keep``      byte ranges (``[(lo, hi), ...]`` or ``"0x...-0x..."``) emitted
                  verbatim.  The trace still steps through them normally, so the
                  declarations keep working, but nothing inside is rewritten - the
                  way to keep a spin lock while using a known lock value to get
                  past it.
    ``protect_loaded_stores``
                  keep a store whose bytes a surviving instruction still reads at
                  runtime.  Only turn this off to reproduce the older, smaller -
                  but unsound - dead-store elimination.
    ``is_end``    callback: True for the stub's last instruction (default: indirect
                  ``jmp`` or ``ret``).
    ``mode``      ``"inplace"`` (patchable, same addresses) or ``"compact"``.
    """
    if mode not in ("inplace", "compact"):
        raise ValueError("mode must be 'inplace' or 'compact'")
    ins_in = _norm_regs(regs_in)
    out = _norm_regs(regs_out)
    init: dict[str, int] = {k: v for k, v in ins_in.items() if v is not None}
    init.update({k: v for k, v in out.items() if v is not None})
    init.update({k: v for k, v in (concrete or {}).items()})
    for name in list(init) + list(ins_in) + list(out):
        if name not in REG32 and name not in FLAGS:
            raise ValueError(f"unknown register {name!r}")

    arg_slots = _norm_args(args)
    mem_slots = _norm_mem(mem)
    trace = analyse(source, start, is_end or is_terminator, concrete=init, args=arg_slots,
                    mem=mem_slots, max_instructions=max_instructions)
    if not trace.steps:
        raise StubError(f"nothing decoded at {start:#x}")
    trace.keep = _read_keep(source, _norm_keep(keep, start), trace.end, mode)
    # slice <-> fold fixpoint: folding turns an instruction into "mov reg, const",
    # which can make earlier instructions dead, which enables more folding
    stats: dict = {}
    for _ in range(8):
        stats = slice_trace(trace, sorted(out), keep_unread_stores=keep_unread_stores,
                            protect_loaded_stores=protect_loaded_stores)
        n = (_fold_pass(trace, stats) if FOLD else 0) + (_drop_esp_runs(trace, stats) if ESP_RUNS else 0)
        if n == 0:
            break                       # slice and passes agree: fixpoint reached
    # the passes have the last word, so recount from the actual kept flags
    stats["ins_kept"] = sum(1 for s in trace.steps if s.kept)
    stats["ins_dropped"] = stats["ins_total"] - stats["ins_kept"]
    stats["folded"] = sum(1 for s in trace.steps if s.kept and s.rewritten)
    stats["absorbed"] = sum(1 for s in trace.steps if s.absorbed)
    stats["esp_runs"] = sum(1 for s in trace.steps if not s.kept and s.esp_dropped)
    stats["protect_loaded_stores"] = protect_loaded_stores
    if arg_slots:
        stats["args"] = ", ".join(f"[esp+{off:#x}]" + (f"={t.val:#x}" if t.op == "c" else "")
                                  for off, t in sorted(arg_slots.items()))
    if mem_slots:
        stats["mem"] = ", ".join(f"[{a:#x}]" + (f"={t.val:#x}" if t.op == "c" else "")
                                 for a, t in sorted(mem_slots.items()))
    if trace.keep:
        stats["keep"] = ", ".join(f"[{lo:#x}-{hi:#x}]" for lo, hi, _, _ in trace.keep)
    code = emit_inplace(trace) if mode == "inplace" else emit_compact(trace)

    outputs = {n: trace.state.regs[n] for n in sorted(out)}
    compare = set(out)
    if trace.terminator is not None:
        for n in sorted(trace.terminator.reads):
            outputs.setdefault(f"{n} (exit)", trace.state.regs[n])
            compare.add(n)
    return Result(start=start, end=trace.end, mode=mode, code=code, trace=trace,
                  stats=stats, outputs=outputs, compare_regs=tuple(sorted(compare)),
                  concrete=dict(init),                 # register values the result assumes
                  concrete_args={off: t.val for off, t in arg_slots.items() if t.op == "c"},
                  concrete_mem={a: t.val for a, t in mem_slots.items() if t.op == "c"})
