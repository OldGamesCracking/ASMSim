"""End-to-end tests for the deobfuscating simplifier.

The real handlers from the snapshot are simplified and the result is executed in
Unicorn side by side with the original: the outputs (EBX and whatever the
terminator reads) and every memory byte outside the stub's private stack frame
must be identical.  A pure-static invariant is checked as well: no kept
instruction may read a register whose last writer was dropped.
"""
import random
from pathlib import Path

import pytest
from iced_x86 import Code, Encoder, Instruction, Mnemonic as MN, OpKind as OK, Register

from asmsim import terms as T
from asmsim import x86asm as asm
from asmsim.deob import (BytesSource, RemoteSource, StubError, analyse, arch_reads, simplify,
                         slice_trace)
from asmsim.emu import Emu, REGS, random_regs
from asmsim.lift import FLAGS, REG32, State
from asmsim.verify import verify

FIXTURES = Path(__file__).parent / "fixtures"
BASE = 0x1141000
HANDLERS = {"handler1": 0x1141AE7, "handler2": 0x11435C8, "handler3": 0x114A3EA}


@pytest.fixture(scope="module")
def src():
    return BytesSource((FIXTURES / "vmcode.bin").read_bytes(), BASE)


@pytest.mark.parametrize("name,addr", sorted(HANDLERS.items()))
def test_simplifies_and_verifies(src, name, addr):
    res = simplify(src, addr, regs_in=["ebx"], regs_out=["ebx"])
    assert res.trace.terminator is not None
    assert res.trace.stop is None
    assert res.stats["ins_kept"] < res.stats["ins_total"] / 2       # a lot must go
    assert res.stats["dead_bytes"] > 0                              # the jmp-over-junk bytes
    assert len(res.code) == res.end - res.start                     # in place
    rep = verify(src, res, runs=40)
    assert rep.ok, f"{name}: {rep}"


@pytest.mark.parametrize("name,addr", sorted(HANDLERS.items()))
def test_compact_mode_verifies(src, name, addr):
    res = simplify(src, addr, regs_in=["ebx"], regs_out=["ebx"], mode="compact")
    assert len(res.code) <= res.end - res.start
    assert res.terminator_ip_out is not None
    rep = verify(src, res, runs=25)
    assert rep.ok, f"{name}: {rep}"


@pytest.mark.parametrize("name,addr", sorted(HANDLERS.items()))
def test_slice_invariant(src, name, addr):
    """A kept instruction must never read a register/flags whose last writer was dropped."""
    trace = analyse(src, addr)
    slice_trace(trace, ["ebx"])
    last: dict = {}
    for step in trace.steps:
        if step.kept:
            reads, flags = arch_reads(step.ins)
            for name_ in sorted(reads | flags):
                if name_ in ("esp", "df"):
                    continue
                writer = last.get(name_)
                assert writer is None or writer.kept, \
                    f"{step.ip:#x} reads {name_} last written by dropped {writer.ip:#x}"
        for name_ in step.writes | step.flag_writes:
            last[name_] = step


def _emitted_reads(ins):
    """``arch_reads`` without the spurious input read of a self-xor."""
    if ins.mnemonic == MN.XOR and ins.op_count == 2 and ins.op_kind(0) == OK.REGISTER \
            and ins.op_kind(1) == OK.REGISTER and ins.op_register(0) == ins.op_register(1):
        return frozenset(), frozenset()
    regs, flags = arch_reads(ins)
    return regs, flags


@pytest.mark.parametrize("name,addr", sorted(HANDLERS.items()))
def test_folded_output_reads_have_kept_writers(src, name, addr):
    """The same invariant on the instructions the fold actually emits."""
    res = simplify(src, addr, regs_in=["ebx"], regs_out=["ebx"])
    last: dict = {}
    for step in res.trace.steps:
        if step.kept:
            for ins in step.instrs():
                reads, flags = _emitted_reads(ins)
                for name_ in sorted(reads | flags):
                    if name_ in ("esp", "df"):
                        continue
                    writer = last.get(name_)
                    assert writer is None or writer.kept, \
                        f"{ins} at {step.ip:#x} reads {name_} last written by dropped {writer.ip:#x}"
        for name_ in step.writes | step.flag_writes:
            last[name_] = step


class _LoopbackClient:
    """Stands in for an attached x64dbg-automate client (page-sized RPC chunks)."""

    def __init__(self, data: bytes, base: int, running: bool = False, fail_writes: bool = False):
        self.data = bytearray(data)
        self.base, self.running, self.fail_writes = base, running, fail_writes
        self.detached = False
        self.reads: list[tuple[int, int]] = []
        self.writes: list[tuple[int, bytes]] = []

    def read_memory(self, addr: int, size: int) -> bytes:
        off = addr - self.base
        if off < 0 or off + size > len(self.data):
            raise RuntimeError("unmapped")
        self.reads.append((addr, size))
        return bytes(self.data[off:off + size])

    def write_memory(self, addr: int, data: bytes) -> bool:
        if self.fail_writes:
            return False
        off = addr - self.base
        if off < 0 or off + len(data) > len(self.data):
            return False
        self.data[off:off + len(data)] = data
        self.writes.append((addr, bytes(data)))
        return True

    def is_running(self) -> bool:
        return self.running

    def detach_session(self) -> None:
        self.detached = True


def test_remote_source_reads_like_the_snapshot():
    data = (FIXTURES / "vmcode.bin").read_bytes()
    client = _LoopbackClient(data, BASE)
    remote = RemoteSource(client)
    snap = BytesSource(data, BASE)
    assert remote.read(0x1141AE7, 16) == snap.read(0x1141AE7, 16)
    assert remote.read(0x1141FFF, 0x1001) == snap.read(0x1141FFF, 0x1001)   # page crossing
    assert len(client.reads) > 1                                            # split at the page edge
    res = simplify(remote, 0x1141AE7, regs_in=["ebx"], regs_out=["ebx"])
    assert res.stats["ins_kept"] < res.stats["ins_total"]
    remote.close()
    assert client.detached


def test_patch_bytes_covers_the_whole_stub_and_verifies():
    """The bytes written over a live stub: the code, NOP padded to start..end."""
    data = (FIXTURES / "vmcode.bin").read_bytes()
    for mode in ("inplace", "compact"):
        res = simplify(BytesSource(data, BASE), 0x1141AE7, regs_in=["ebx"], regs_out=["ebx"],
                       mode=mode)
        patch = res.patch_bytes()
        assert len(patch) == res.end - res.start          # no stale bytes left behind
        assert patch.startswith(res.code)
        assert set(patch[len(res.code):]) <= {0x90}       # the rest is NOPs
        if mode == "inplace":
            assert patch == res.code


def test_live_patch_writes_the_blob_and_reads_it_back():
    """Patching through a (fake, paused) x64dbg session replaces the stub in place."""
    from tools.deob import patch_live

    data = (FIXTURES / "vmcode.bin").read_bytes()
    client = _LoopbackClient(data, BASE)
    remote = RemoteSource(client)
    res = simplify(remote, 0x1141AE7, regs_in=["ebx"], regs_out=["ebx"])
    assert patch_live(remote, res) is None
    off = res.start - BASE
    assert bytes(client.data[off:off + len(res.patch_bytes())]) == res.patch_bytes()

    # ... and it refuses while the target is running, or on a failed write
    running = RemoteSource(_LoopbackClient(data, BASE, running=True))
    assert "running" in patch_live(running, res)
    broken = RemoteSource(_LoopbackClient(data, BASE, fail_writes=True))
    assert "failed" in patch_live(broken, res)


def test_concrete_input_registers_fold():
    """A concrete RegsIn value is substituted, so the outputs become constants."""
    data = (FIXTURES / "vmcode.bin").read_bytes()
    snap = BytesSource(data, BASE)
    res = simplify(snap, 0x1141AE7, regs_in={"ebx": 0x1000}, regs_out=["ebx"])
    assert res.outputs["ebx"].op == "c"
    assert res.outputs["ebx"].args[0] == 0x1000                 # EBX is passed through unchanged


def test_patched_image_contains_the_simplified_stub(src):
    """The inplace blob drops straight into the original image at the same address."""
    res = simplify(src, 0x1141AE7, regs_in=["ebx"], regs_out=["ebx"])
    image = bytearray(src.data)
    off = 0x1141AE7 - BASE
    image[off:off + len(res.code)] = res.code
    patched = BytesSource(bytes(image), BASE)
    assert patched.read(0x1141AE7, len(res.code)) == res.code
    rep = verify(patched, res, runs=20)          # the patched image behaves like the original
    assert rep.ok, str(rep)


def test_inplace_dropped_bytes_are_nops(src):
    res = simplify(src, 0x1141AE7, regs_in=["ebx"], regs_out=["ebx"])
    covered = set()
    for step in res.kept_steps:
        covered.update(range(step.ip, step.region_end))
    for i, byte in enumerate(res.code):
        addr = res.start + i
        if addr not in covered:
            assert byte == 0x90, f"{addr:#x} was not kept and is not a NOP"


# ===========================================================================================
# run folding: chains of instructions that end in a literal
# ===========================================================================================
STUB_BASE = 0x400000


def _stub(insns):
    """Assemble a straight-line stub, padded to a full page so ``verify`` can read it."""
    data = asm.encode(list(insns), STUB_BASE) + b"\x90" * 32
    return BytesSource(data + b"\x90" * (0x1000 - len(data)), STUB_BASE)


def _jmp(reg_name: str):
    return Instruction.create_reg(Code.JMP_RM32, getattr(Register, reg_name.upper()))


def test_constant_run_folds_to_a_literal():
    """MOV EAX,1 / SHL EAX,2 -> MOV EAX,4 and MOV CL,0x22 / XOR CL,0x3A -> MOV CL,0x18."""
    src = _stub([
        asm.mov(asm.reg("eax"), asm.imm(1)),
        asm.shift("shl", asm.reg("eax"), asm.imm(2)),
        asm.mov(asm.reg("ecx", 8), asm.imm(0x22)),
        asm.alu("xor", asm.reg("ecx", 8), asm.imm(0x3A)),
        _jmp("eax"),
    ])
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax", "ecx"])
    assert res.code[:7] == bytes.fromhex("b804000000b118")
    assert res.code[7:12] == b"\x90" * 5             # the rest of the run is padding
    assert res.stats["absorbed"] == 3
    assert res.stats["ins_kept"] == 2                # the folded run and the jmp
    # a folded run is listed one instruction per line, each at its own address
    folded = [l for l in res.listing().splitlines() if "mov eax, 4" in l or "mov cl, 0x18" in l]
    assert len(folded) == 2
    assert "mov cl" not in folded[0] and "mov eax" not in folded[1]
    rep = verify(src, res, runs=25)
    assert rep.ok, str(rep)


def test_partial_writes_fold_to_their_register_view():
    """A run that ends in a 16 bit / high byte literal is written back as such."""
    src = _stub([
        asm.mov(asm.reg("ecx", 16), asm.imm(1)),
        asm.alu("or", asm.reg("ecx", 16), asm.imm(2)),
        asm.mov(asm.reg("eax", "h"), asm.imm(0x10)),
        asm.alu("xor", asm.reg("eax", "h"), asm.imm(1)),
        _jmp("ebx"),
    ])
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax", "ecx"])
    listing = res.listing()
    assert "mov cx, 3" in listing and "mov ah, 0x11" in listing
    assert res.stats["absorbed"] == 3                # the run folds both registers at once
    rep = verify(src, res, runs=25)
    assert rep.ok, str(rep)


def test_run_folds_when_a_later_instruction_overwrites_the_flags():
    """The XOR's flags are dead (the ADD overwrites them), so the run may fold."""
    src = _stub([
        asm.mov(asm.reg("ecx", 8), asm.imm(0x22)),
        asm.alu("xor", asm.reg("ecx", 8), asm.imm(0x3A)),
        asm.alu("add", asm.reg("eax"), asm.imm(1)),
        asm.setcc("e", asm.reg("ebx", 8)),           # reads ZF - from the ADD, not the XOR
        _jmp("eax"),
    ])
    res = simplify(src, STUB_BASE, regs_in=["eax", "ebx"], regs_out=["eax", "ecx", "ebx"])
    listing = res.listing()
    assert "mov cl, 0x18" in listing
    assert "add eax, 1" in listing and "set" in listing
    rep = verify(src, res, runs=25)
    assert rep.ok, str(rep)


def test_run_stays_when_a_later_instruction_reads_its_flags():
    """SETE BL reads the XOR's ZF: folding it away would change BL."""
    src = _stub([
        asm.mov(asm.reg("ecx", 8), asm.imm(0x22)),
        asm.alu("xor", asm.reg("ecx", 8), asm.imm(0x3A)),
        asm.setcc("e", asm.reg("ebx", 8)),
        _jmp("eax"),
    ])
    res = simplify(src, STUB_BASE, regs_in=["eax", "ebx"], regs_out=["eax", "ecx", "ebx"])
    assert "xor cl, 0x3a" in res.listing().lower()
    rep = verify(src, res, runs=25)
    assert rep.ok, str(rep)


def _raw_stub(hexstr: str):
    """A stub from hand written bytes (regression reproducers), NOP padded."""
    data = bytes.fromhex(hexstr) + b"\x90" * 16
    return BytesSource(data + b"\x90" * (0x1000 - len(data)), STUB_BASE)


def _assert_regions_do_not_overlap(res):
    spans = sorted((s.ip, s.region_end) for s in res.kept_steps)
    for (lo, hi), (lo2, _hi2) in zip(spans, spans[1:]):
        assert hi <= lo2, f"a folded run overlaps the next step: {lo:#x}..{hi:#x} / {lo2:#x}"


def test_step_dropped_inside_a_run_cannot_come_back():
    """``dec esi`` is dead while the run folds, but live again afterwards; it has
    to stay absorbed - re-keeping it would overwrite the replacement bytes."""
    hexs = ("89c666be85c4"    # mov esi,eax ; mov si,0xc485
            "6683c15f"        # add cx,0x5f
            "685d560000"      # push 0x565d
            "b122"            # mov cl,0x22
            "ffce"            # dec esi          <- dropped, then live again
            "be02000000"      # mov esi,2
            "b410"            # mov ah,0x10
            "b510"            # mov ch,0x10
            "6878563412"      # push 0x12345678
            "ffe3")           # jmp ebx
    src = _raw_stub(hexs)
    res = simplify(src, STUB_BASE, regs_in=["ebx"],
                   regs_out=["eax", "ecx", "edx", "ebx", "esi", "edi", "ebp"])
    _assert_regions_do_not_overlap(res)
    rep = verify(src, res, runs=20)
    assert rep.ok, str(rep)


def test_partial_write_keeps_the_producer_of_the_bits_it_merges():
    """``mov bl, 0`` keeps ebx's upper bits, so ``mov ebx, ebp`` - the producer of
    the value they merge into - has to stay in the output."""
    hexs = ("6a7f6a7f"        # push 0x7f ; push 0x7f
            "89eb"            # mov ebx,ebp       <- must survive the fold of "sete bl"
            "b122"            # mov cl,0x22
            "6683e93f"        # sub cx,0x3f
            "0f94c3"          # sete bl
            "89f1"            # mov ecx,esi
            "80c2ff"          # add dl,0xff
            "66be6ea9"        # mov si,0xa96e
            "80c101"          # add cl,1
            "ffe6")           # jmp esi
    src = _raw_stub(hexs)
    res = simplify(src, STUB_BASE, regs_in=["ebp"],
                   regs_out=["eax", "ecx", "edx", "ebx", "esi", "edi"])
    assert "mov ebx, ebp" in res.listing()
    _assert_regions_do_not_overlap(res)
    rep = verify(src, res, runs=20)
    assert rep.ok, str(rep)


def test_a_store_a_surviving_load_reads_stays():
    """``pop esi`` reads [esp] back at runtime: the spill it forwards from must stay
    in the output even though ESI's *symbolic* value does not change."""
    hexs = ("89edb122"        # mov ebp,ebp ; mov cl,0x22
            "893424"          # mov [esp],esi   <- re-read by the pop below
            "0f95c3"          # setne bl
            "5e"              # pop esi
            "21de80c318"      # and esi,ebx ; add bl,0x18
            "6683f257"        # xor dx,0x57
            "6a7f"            # push 0x7f        <- overwrites the same slot
            "f7d2"            # not edx
            "66bddba9"        # mov bp,0xa9db
            "891dfcffffff"    # mov [0xfffffffc],ebx
            "ffe6")           # jmp esi
    src = _raw_stub(hexs)
    outs = ["eax", "ecx", "edx", "ebx", "esi", "edi", "ebp"]
    res = simplify(src, STUB_BASE, regs_in=[], regs_out=outs)
    assert "mov [esp], esi" in res.listing()
    _assert_regions_do_not_overlap(res)
    rep = verify(src, res, runs=25)
    assert rep.ok, str(rep)

    # ... and the switch really is the difference: without the protection the spill
    # disappears (that is the unsound behaviour the option restores)
    loose = simplify(src, STUB_BASE, regs_in=[], regs_out=outs, protect_loaded_stores=False)
    assert "mov [esp], esi" not in loose.listing(show_dropped=False)
    assert not verify(src, loose, runs=5).ok


def test_loaded_store_protection_can_be_switched_off():
    """protect_loaded_stores=False (``--unsafe-stores``) goes back to the code the
    simplifier emitted before the fix."""
    src = _stub([
        asm.mov(asm.mem(asm.Mem(base="esp")), asm.imm(0x12345678)),
        asm.push(asm.imm(0x22334455)),
        asm.mov(asm.reg("eax"), asm.mem(asm.Mem(base="esp"))),
        _jmp("eax"),
    ])
    on = simplify(src, STUB_BASE, regs_in=[], regs_out=["eax", "ecx"])
    off = simplify(src, STUB_BASE, regs_in=[], regs_out=["eax", "ecx"],
                   protect_loaded_stores=False)
    assert on.stats["protect_loaded_stores"] is True
    assert off.stats["protect_loaded_stores"] is False
    assert "--unsafe-stores" in off.summary()
    for res in (on, off):
        assert verify(src, res, runs=15).ok, res.listing()


def test_constant_stack_idiom_still_folds():
    """The VM's ``sub esp,4 / mov [esp],imm / pop reg`` must still collapse into one
    literal write even though the load now keeps its store alive."""
    src = _stub([
        asm.alu("sub", asm.reg("esp"), asm.imm(4)),
        asm.mov(asm.mem(asm.Mem(base="esp")), asm.imm(0x45505948)),
        Instruction.create_reg(Code.POP_R32, Register.EAX),
        _jmp("eax"),
    ])
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"])
    assert res.stats["ins_kept"] == 2                # the folded run and the jmp
    assert "mov eax, 0x45505948" in res.listing()
    rep = verify(src, res, runs=15)
    assert rep.ok, str(rep)


# ===========================================================================================
# stack arguments: --args arg0, arg1, ... at [esp+0], [esp+4], ...
# ===========================================================================================
def _arg_stub():
    return _stub([
        asm.mov(asm.reg("eax"), asm.mem(asm.Mem(base="esp"))),
        asm.alu("add", asm.reg("eax"), asm.mem(asm.Mem(base="esp", disp=4))),
        _jmp("eax"),
    ])


def test_stack_args_become_named_inputs():
    res = simplify(_arg_stub(), STUB_BASE, regs_in=[], regs_out=["eax"],
                   args=["arg0", "arg1"])
    assert T.fmt(res.outputs["eax"], 8) == "add(in:arg0, in:arg1)"
    assert "in:arg0" in res.listing()
    assert res.stats["args"] == "[esp+0x0], [esp+0x4]"
    assert res.concrete_args == {}
    rep = verify(_arg_stub(), res, runs=20)
    assert rep.ok, str(rep)


def test_concrete_stack_args_fold_to_literals():
    res = simplify(_arg_stub(), STUB_BASE, regs_in=[], regs_out=["eax"],
                   args={"arg0": 0x10, "arg1": 2})
    assert res.outputs["eax"].op == "c" and res.outputs["eax"].args[0] == 0x12
    assert "mov eax, 0x12" in res.listing()
    assert res.concrete_args == {0: 0x10, 4: 2}
    rep = verify(_arg_stub(), res, runs=20)      # the verifier seeds both slots
    assert rep.ok, str(rep)


def test_a_store_to_an_arg_slot_wins():
    """A declared argument must not shadow what the stub itself stored there."""
    src = _stub([
        asm.mov(asm.mem(asm.Mem(base="esp")), asm.imm(5)),
        asm.mov(asm.reg("eax"), asm.mem(asm.Mem(base="esp"))),
        _jmp("eax"),
    ])
    res = simplify(src, STUB_BASE, regs_in=[], regs_out=["eax"], args={"arg0": 0x11223344})
    assert res.outputs["eax"].op == "c" and res.outputs["eax"].args[0] == 5
    rep = verify(src, res, runs=15)
    assert rep.ok, str(rep)


def test_a_read_inside_an_arg_slot_is_a_slice_of_it():
    src = _stub([
        asm.movx("movzx", "eax", asm.mem(asm.Mem(base="esp", disp=1), 8)),
        asm.movx("movzx", "ecx", asm.mem(asm.Mem(base="esp", disp=2), 16)),
        _jmp("eax"),
    ])
    res = simplify(src, STUB_BASE, regs_in=[], regs_out=["eax", "ecx"],
                   args={"arg0": 0xAABBCCDD})
    assert res.outputs["eax"].args[0] == 0xCC          # byte 1 of the little endian dword
    assert res.outputs["ecx"].args[0] == 0xAABB        # its high word
    rep = verify(src, res, runs=15)
    assert rep.ok, str(rep)


def test_reads_outside_or_across_slots_stay_opaque():
    src = _stub([asm.mov(asm.reg("eax"), asm.mem(asm.Mem(base="esp", disp=2))), _jmp("eax")])
    res = simplify(src, STUB_BASE, regs_in=[], regs_out=["eax"], args=["arg0", "arg1"])
    assert res.outputs["eax"].op == "load"             # spans arg0 and arg1

    below = _stub([asm.mov(asm.reg("eax"), asm.mem(asm.Mem(base="esp", disp=-4))), _jmp("eax")])
    res = simplify(below, STUB_BASE, regs_in=[], regs_out=["eax"], args=["arg0"])
    assert res.outputs["eax"].op == "load"             # below the first slot


def test_declared_but_unread_args_change_nothing():
    src = _stub([asm.mov(asm.reg("eax"), asm.imm(7)), _jmp("eax")])
    plain = simplify(src, STUB_BASE, regs_in=[], regs_out=["eax"])
    withargs = simplify(src, STUB_BASE, regs_in=[], regs_out=["eax"], args=["arg0", "arg1"])
    assert plain.code == withargs.code
    assert withargs.stats["args"] == "[esp+0x0], [esp+0x4]"


def test_a_terminator_reading_an_arg_slot_is_handled():
    """``ret`` / ``jmp [esp]`` take the argument at [esp]: the name must not leak
    into the compared registers (collecting the outputs used to raise a KeyError)."""
    for hexs in ("ff2424",      # jmp dword ptr [esp]
                 "c3"):         # ret
        res = simplify(_raw_stub(hexs), STUB_BASE, regs_in=[], regs_out=["eax"], args=["arg0"])
        assert res.trace.terminator is not None
        assert "arg0" not in res.compare_regs
        assert res.outputs["eax"].op == "in"


def test_stack_argument_names_are_validated():
    src = _stub([asm.mov(asm.reg("eax"), asm.imm(1)), _jmp("eax")])
    for bad in (["eax"], ["arg"], ["argx"], ["0"], ["arg0", "arg00"]):
        with pytest.raises(ValueError):
            simplify(src, STUB_BASE, regs_in=[], regs_out=["eax"], args=bad)


def test_argument_slots_survive_a_state_copy():
    st = State.initial(args={0: T.inp("arg0", 32)})
    assert st.mem.arg_slots == {0: T.inp("arg0", 32)}
    assert st.copy().mem.arg_slots == st.mem.arg_slots


def test_parse_regs_accepts_the_argument_syntax():
    from tools.deob import parse_regs
    assert parse_regs("arg0,arg1=0x1234") == {"arg0": None, "arg1": 0x1234}


# ===========================================================================================
# known memory contents (--mem address=value) and the instructions they unlock
# ===========================================================================================
DATA = 0x600100          # lazily mapped data, outside the code image


def _branchy_stub():
    """``mov al, [DATA] / cmp al, 7 / jne end / mov eax, 0x111 / end: jmp ebx``."""
    head = [
        asm.mov(asm.reg("eax", 8), asm.mem(asm.Mem(disp=DATA), 8)),
        asm.alu("cmp", asm.reg("eax", 8), asm.imm(7), 8),
    ]
    size = sum(len(asm.encode([i], STUB_BASE)) for i in head)
    jne = asm.jcc("ne", STUB_BASE + size + 5)        # past the "mov eax, 0x111"
    return _stub(head + [jne, asm.mov(asm.reg("eax"), asm.imm(0x111)), _jmp("ebx")])


def test_known_memory_resolves_a_data_dependent_branch():
    src = _branchy_stub()
    taken = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"], mem={DATA: 7})
    assert taken.trace.stop is None                  # the branch became a plain jump
    assert taken.outputs["eax"].op == "c" and taken.outputs["eax"].args[0] == 0x111
    assert taken.stats["mem"] == f"[{DATA:#x}]=0x7"
    assert verify(src, taken, runs=20).ok            # the verifier seeds the value

    other = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"], mem={"0x600100": 9})
    # the other edge: the branch is taken, only the loaded byte was written
    assert other.outputs["eax"].op == "ins"
    assert verify(src, other, runs=20).ok


def test_several_memory_declarations_and_their_validation():
    """--mem takes a comma separated list; neighbouring dwords are fine, overlapping
    or repeated ones are not."""
    src = _branchy_stub()
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"],
                   mem={"0x600100": 7, "0x600104": 0x99})
    assert res.stats["mem"] == "[0x600100]=0x7, [0x600104]=0x99"
    assert res.concrete_mem == {0x600100: 7, 0x600104: 0x99}
    assert verify(src, res, runs=15).ok

    with pytest.raises(ValueError):                 # overlapping dwords
        simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"],
                 mem={"0x600100": 1, "0x600102": 2})
    with pytest.raises(ValueError):                 # the same address listed twice
        simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"],
                 mem=["0x600100", "0x600100"])


KEEP_STUB = ("8b0424"            # mov eax, [esp]          (arg0)
             "83c001"            # add eax, 1        <- region
             "eb03"              # jmp short +3      <- region
             "909090"            # ...junk           <- region
             "03442404"          # add eax, [esp+4]        (arg1)
             "ffe3")             # jmp ebx
KEEP_LO, KEEP_HI = STUB_BASE + 3, STUB_BASE + 0x0b


def test_a_kept_region_is_emitted_verbatim():
    src = _raw_stub(KEEP_STUB)
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"],
                   args=["arg0", "arg1"], keep=[(KEEP_LO, KEEP_HI)])
    # the region's trace ran (its `add eax, 1` is in the value) ...
    assert T.fmt(res.outputs["eax"], 8) == "add(in:arg1, add(in:arg0, 1))"
    # ... but its bytes - including the junk the jump skips - are untouched
    assert res.code[3:0x0b] == bytes.fromhex("83c001" "eb03" "909090")
    assert res.stats["keep"] == f"[{KEEP_LO:#x}-{KEEP_HI:#x}]"
    listing = res.listing()
    assert f"; --- kept verbatim {KEEP_LO:08X}..{KEEP_HI:08X} ---" in listing
    assert "jmp short" in listing.lower()             # the region's own branch, verbatim
    rep = verify(src, res, runs=20)
    assert rep.ok, str(rep)


def test_a_kept_region_also_works_compacted():
    src = _raw_stub(KEEP_STUB)
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"],
                   args=["arg0", "arg1"], keep=[(KEEP_LO, KEEP_HI)], mode="compact")
    assert "jmp short" in res.listing().lower()
    rep = verify(src, res, runs=20)
    assert rep.ok, str(rep)


def test_a_kept_region_branching_out_of_itself_is_refused_compacted():
    """A compact blob moves code, so a branch that leaves its region cannot be kept."""
    # mov eax, [esp] ; jmp short 0x400007 (out of the region) ; nop nop ; jmp ebx
    src = _raw_stub("8b0424" + "eb02" + "9090" + "ffe3")
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"],
                   keep=[(STUB_BASE + 3, STUB_BASE + 5)])          # inplace: fine
    assert verify(src, res, runs=10).ok
    with pytest.raises(ValueError):
        simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"],
                 keep=[(STUB_BASE + 3, STUB_BASE + 5)], mode="compact")


def test_keep_region_validation():
    src = _raw_stub(KEEP_STUB)
    for bad in ("0x1",                      # not LO-HI
                f"{STUB_BASE - 0x10:#x}-{STUB_BASE:#x}",            # starts before the stub
                f"{KEEP_LO:#x}-{KEEP_LO + 2:#x},{KEEP_HI - 2:#x}-{KEEP_HI:#x}"):  # overlap
        with pytest.raises(ValueError):
            simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"], keep=bad)


def test_lock_prefix_is_lifted_as_a_plain_instruction():
    """One thread, no memory ordering: ``lock`` adds nothing to the model."""
    src = _raw_stub("f001442408"                      # lock add dword [esp+8], eax
                    "ffe3")                           # jmp ebx
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"])
    assert res.trace.stop is None
    assert "lock add" in res.listing()
    rep = verify(src, res, runs=20)                   # the store is at [esp+8]: compared
    assert rep.ok, str(rep)


def test_rep_stosd_is_unrolled_into_stores():
    """A constant count ``rep stos`` becomes the stores it performs; the instruction
    itself stays in the output, so the blob keeps its shape."""
    src = _raw_stub("8bfc"                            # mov edi, esp
                    "b844332211"                      # mov eax, 0x11223344
                    "b903000000"                      # mov ecx, 3
                    "f3ab"                            # rep stosd
                    "ffe3")                           # jmp ebx
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"])
    stores = [s for s in res.trace.state.mem.stores]
    assert [T.fmt(s.val) for s in stores] == ["0x11223344"] * 3
    assert [T.fmt(s.addr, 3) for s in stores] == ["in:esp", "add(in:esp, 4)",
                                                 "add(in:esp, 8)"]
    assert "rep stosd" in res.listing()
    rep = verify(src, res, runs=20)                   # the writes land at [esp..esp+12)
    assert rep.ok, str(rep)


def test_a_symbolic_rep_count_is_not_lifted():
    """Without a known count the unrolled stores cannot be built - the analysis stops."""
    src = _raw_stub("8bfc" + "b844332211" + "f3ab" + "ffe3")   # ecx never written
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"])
    assert res.trace.stop is not None
    assert "symbolic count" in res.trace.stop.why


def test_computed_values_are_left_alone():
    """Nothing to fold when the value is not a literal."""
    src = _stub([asm.alu("add", asm.reg("eax"), asm.reg("ebx")), _jmp("eax")])
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"])
    assert "add eax, ebx" in res.listing()
    assert res.stats["absorbed"] == 0
    rep = verify(src, res, runs=15)
    assert rep.ok, str(rep)


# ===========================================================================================
# conditional branches: synthetic shapes
# ===========================================================================================
class _Asm:
    """A tiny assembler with labels.  Branches are always the near form (fixed size),
    so one sizing pass gives exact addresses and nothing is relaxed afterwards."""

    def __init__(self, base: int, data: bytes = b""):
        self.base = base
        self.parts: list = []
        self.data = data

    def label(self, name: str) -> "_Asm":
        self.parts.append(("label", name))
        return self

    def op(self, *instrs) -> "_Asm":
        self.parts.append(("ins", list(instrs)))
        return self

    def jcc(self, cc: str, label: str) -> "_Asm":
        self.parts.append(("jcc", (cc, label)))
        return self

    def jmp(self, label: str) -> "_Asm":
        self.parts.append(("jmp", label))
        return self

    @staticmethod
    def _enc(ins, ip: int) -> bytes:
        enc = Encoder(32)
        enc.encode(ins, ip)
        return bytes(enc.take_buffer())

    def _length(self, ins, ip: int) -> int:
        return len(self._enc(ins, ip))

    def build(self) -> bytes:
        addr: dict[str, int] = {}
        ip = self.base
        for kind, payload in self.parts:
            if kind == "label":
                addr[payload] = ip
            elif kind == "ins":
                for ins in payload:
                    ip += self._length(ins, ip)
            else:
                ip += 6 if kind == "jcc" else 5
        out, ip = bytearray(), self.base
        for kind, payload in self.parts:
            if kind == "label":
                continue
            if kind == "ins":
                for ins in payload:
                    out += self._enc(ins, ip)
                    ip += self._length(ins, ip)
            elif kind == "jcc":
                cc, label = payload
                out += self._enc(asm.jcc(cc, addr[label]), ip)
                ip += 6
            else:
                out += self._enc(asm.jmp(addr[payload]), ip)
                ip += 5
        return bytes(out)


def _estub(code: bytes, base: int = STUB_BASE):
    data = code + b"\x90" * 16
    return BytesSource(data + b"\x90" * (0x1000 - len(data)), base)


P = 0x600200        # an absolute scratch address (observable: above the entry ESP)


def _diamond_stub():
    a = _Asm(STUB_BASE)
    a.op(asm.alu("cmp", asm.reg("eax"), asm.imm(7)))
    a.jcc("ne", "else")
    a.op(asm.mov(asm.reg("ecx"), asm.imm(5))).jmp("end")
    a.label("else")
    a.op(asm.mov(asm.reg("ecx"), asm.imm(7)))
    a.label("end")
    a.op(_jmp("ebx"))
    return _estub(a.build())


def test_a_diamond_merges_its_arms():
    src = _diamond_stub()
    res = simplify(src, STUB_BASE, regs_in=["eax", "ebx"], regs_out=["ecx"])
    assert res.trace.stop is None
    assert res.outputs["ecx"].op == "ite"                 # the merge
    assert sum(1 for s in res.kept_steps if s.kind == "jcc") == 1
    rep = verify(src, res, runs=30)
    assert rep.ok, str(rep)


def _store_diamond_stub(second_store: bool = True):
    a = _Asm(STUB_BASE)
    a.op(asm.alu("cmp", asm.reg("eax"), asm.imm(7)))
    a.jcc("ne", "else")
    a.op(asm.mov(asm.mem(asm.Mem(disp=P)), asm.imm(0x11))).jmp("end")
    a.label("else")
    if second_store:
        a.op(asm.mov(asm.mem(asm.Mem(disp=P)), asm.imm(0x22)))
    a.label("end")
    a.op(_jmp("ebx"))
    return _estub(a.build())


def test_stores_on_both_arms_both_survive_and_verify():
    """A store in one arm is not "covered" by a store in the other - DSE stays per arm."""
    src = _store_diamond_stub()
    res = simplify(src, STUB_BASE, regs_in=["eax", "ebx"], regs_out=[])
    stores = [s for s in res.kept_steps if s.stores]
    assert len(stores) == 2
    rep = verify(src, res, runs=40)
    assert rep.ok, str(rep)


def test_a_load_after_a_rejoin_keeps_the_store_it_may_read():
    """The load after the rejoin can read what the taken arm stored: it has to stay."""
    src = _store_diamond_stub(second_store=False)
    res = simplify(src, STUB_BASE, regs_in=["eax", "ebx"], regs_out=["ecx"])
    assert any(s.stores for s in res.kept_steps)
    rep = verify(src, res, runs=40)
    assert rep.ok, str(rep)


def _nested_diamond_stub():
    a = _Asm(STUB_BASE)
    a.op(asm.alu("cmp", asm.reg("eax"), asm.imm(7)))
    a.jcc("ne", "else1")
    a.op(asm.alu("cmp", asm.reg("edx"), asm.imm(9)))
    a.jcc("ne", "else2")
    a.op(asm.mov(asm.reg("ecx"), asm.imm(1))).jmp("inner")
    a.label("else2")
    a.op(asm.mov(asm.reg("ecx"), asm.imm(2)))
    a.label("inner")
    a.jmp("end")
    a.label("else1")
    a.op(asm.mov(asm.reg("ecx"), asm.imm(3)))
    a.label("end")
    a.op(_jmp("ebx"))
    return _estub(a.build())


def test_nested_diamonds():
    src = _nested_diamond_stub()
    res = simplify(src, STUB_BASE, regs_in=["eax", "edx", "ebx"], regs_out=["ecx"])
    assert res.trace.stop is None
    assert sum(1 for s in res.kept_steps if s.kind == "jcc") == 2
    rep = verify(src, res, runs=40)
    assert rep.ok, str(rep)


def _two_exit_stub():
    a = _Asm(STUB_BASE)
    a.op(asm.alu("test", asm.reg("eax", 8), asm.imm(1), 8))
    a.jcc("ne", "other")
    a.op(asm.mov(asm.reg("eax"), asm.imm(0x111)), _jmp("ebx"))
    a.label("other")
    a.op(asm.mov(asm.reg("eax"), asm.imm(0x222)),
         Instruction.create(Code.RETND))
    return _estub(a.build())


def test_an_arm_that_exits_gives_a_second_exit():
    for mode in ("inplace", "compact"):
        src = _two_exit_stub()
        res = simplify(src, STUB_BASE, regs_in=["eax", "ebx"], regs_out=["eax"], mode=mode)
        assert len(res.exits) == 2
        assert all(e.out_ip is not None for e in res.exits)
        assert any("@" in k for k in res.outputs)         # per-exit values
        rep = verify(src, res, runs=60)
        assert rep.ok, str(rep)
        assert all(v > 0 for v in rep.coverage.values()), str(rep)   # both exits taken


def _mem_diamond_stub():
    a = _Asm(STUB_BASE)
    a.op(asm.mov(asm.reg("eax", 8), asm.mem(asm.Mem(disp=DATA), 8)))
    a.op(asm.alu("cmp", asm.reg("eax", 8), asm.imm(7), 8))
    a.jcc("ne", "other")
    a.op(asm.mov(asm.reg("eax"), asm.imm(0x111)), _jmp("ebx"))
    a.label("other")
    a.op(asm.mov(asm.reg("eax"), asm.imm(0x222)), _jmp("ebx"))
    return _estub(a.build())


def test_a_constant_condition_removes_the_dead_arm():
    src = _mem_diamond_stub()
    taken_fall = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"], mem={DATA: 7})
    assert taken_fall.trace.stop is None
    assert taken_fall.outputs["eax"].op == "c" and taken_fall.outputs["eax"].args[0] == 0x111
    assert not any(s.kind == "jcc" for s in taken_fall.kept_steps)
    assert verify(src, taken_fall, runs=20).ok
    taken_jump = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax"], mem={DATA: 9})
    assert taken_jump.outputs["eax"].op == "c" and taken_jump.outputs["eax"].args[0] == 0x222
    assert verify(src, taken_jump, runs=20).ok


def test_a_fold_reaches_across_dead_instructions():
    """``mov cl, x`` / dead junk / ``xor cl, y`` collapses into one literal write.

    The instruction in between is dropped, but it still sits between the two writes
    in the byte stream: the run has to reach across it for the first write to die.
    """
    a = _Asm(STUB_BASE)
    a.op(asm.mov(asm.reg("ecx", 8), asm.imm(0x22)))
    a.op(asm.alu("xor", asm.reg("esi"), asm.reg("edi")))     # only dead registers
    a.op(asm.alu("xor", asm.reg("ecx", 8), asm.imm(0x3A)))
    a.op(_jmp("ebx"))
    src = _estub(a.build())
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["ecx"])
    moves = [asm.fmt(i) for s in res.kept_steps for i in s.instrs()
             if asm.fmt(i).startswith("mov cl")]
    assert moves == ["mov cl, 0x18"], res.listing()
    rep = verify(src, res, runs=20)
    assert rep.ok, str(rep)


def _instructions_only(res) -> str:
    """The listing's instructions, without the annotations (they quote the originals)."""
    kept = res.listing(show_dropped=False)
    return "\n".join(line.split(";")[0] for line in kept.splitlines())


def test_an_arithmetic_chain_folds_its_constants():
    """``mov edi, 0x15`` / ``add edi, ebx`` / ``sub edi, 5`` -> ``mov edi, ebx`` /
    ``add edi, 0x10``: the constants are folded together across the chain."""
    a = _Asm(STUB_BASE)
    a.op(asm.mov(asm.reg("edi"), asm.imm(0x15)))
    a.op(asm.alu("add", asm.reg("edi"), asm.reg("ebx")))
    a.op(asm.alu("sub", asm.reg("edi"), asm.imm(5)))
    a.op(_jmp("eax"))
    src = _estub(a.build())
    res = simplify(src, STUB_BASE, regs_in=["eax", "ebx"], regs_out=["edi"])
    text = _instructions_only(res)                      # the notes mention the originals
    assert "mov edi, ebx" in text and "add edi, 0x10" in text
    assert "sub edi, 5" not in text and "mov edi, 0x15" not in text
    rep = verify(src, res, runs=30)
    assert rep.ok, str(rep)


def test_an_arithmetic_chain_keeps_its_flags_when_they_are_used():
    """The same chain, but a later ``sete`` reads the ``sub``'s ZF: it must survive.

    The ``mov``/``add`` pair may still fold (it keeps the very same sum, so the very
    same flags), but the ``sub`` is what sets the flags the ``sete`` consumes, so the
    run that would fold it away has to be refused.
    """
    a = _Asm(STUB_BASE)
    a.op(asm.mov(asm.reg("edi"), asm.imm(0x15)))
    a.op(asm.alu("add", asm.reg("edi"), asm.reg("ebx")))
    a.op(asm.alu("sub", asm.reg("edi"), asm.imm(5)))
    a.op(asm.setcc("e", asm.reg("ecx", 8)))                   # reads ZF
    a.op(_jmp("eax"))
    src = _estub(a.build())
    res = simplify(src, STUB_BASE, regs_in=["eax", "ebx"], regs_out=["edi", "ecx"])
    text = _instructions_only(res)
    assert "sub edi, 5" in text and "sete cl" in text
    rep = verify(src, res, runs=30)                          # cl is an output: ZF compared
    assert rep.ok, str(rep)


def test_a_fold_across_dead_code_still_respects_the_flags():
    """A run whose ``xor`` sets a *needed* flag must not collapse into a bare mov.

    Same shape as the handler's ``mov cl, x`` / dead junk / ``xor cl, y`` - but the
    low byte comes from a register (so the flag is not statically known) and a later
    ``sete`` reads ZF: replacing the run would drop that definition, so it has to be
    refused.  (With a literal first write the flag *is* known - ZF == 0 - and the
    pass instead folds the ``sete`` to ``mov bl, 0`` and then the run, which is fine.)
    """
    a = _Asm(STUB_BASE)
    a.op(asm.mov(asm.reg("ecx", 8), asm.reg("edx", 8)))      # symbolic low byte
    a.op(asm.alu("xor", asm.reg("esi"), asm.reg("edi")))     # dead junk in between
    a.op(asm.alu("xor", asm.reg("ecx", 8), asm.imm(0x3A)))
    a.op(asm.setcc("e", asm.reg("ebx", 8)))                  # reads ZF
    a.op(_jmp("eax"))
    src = _estub(a.build())
    res = simplify(src, STUB_BASE, regs_in=["eax"], regs_out=["ecx", "ebx"])
    listing = res.listing().lower()
    assert "mov cl, dl" in listing                       # the first write survives
    assert "xor cl, 0x3a" in listing                     # ... and so does the xor
    assert not any(s.rewritten for s in res.kept_steps)
    rep = verify(src, res, runs=30)                      # bl is an output: ZF is compared
    assert rep.ok, str(rep)


def test_segment_register_filler_is_tolerated():
    """``push ss`` / ``pop ss`` junk - the VM sprinkles it in, and the fixture's
    0x114A8EB stops dead on it without support.

    The selector is never modelled (pushing it pushes an unknown, popping it changes
    nothing that is modelled), so all that is left of the pair is its stack effect -
    which the lifter models and the reference emulator performs by hand, since unicorn
    cannot load a segment register at all.
    """
    a = _Asm(STUB_BASE)
    a.op(asm.mov(asm.reg("ecx"), asm.imm(0x1234)))
    a.op(Instruction.create_reg(Code.PUSHD_SS, Register.SS),
         Instruction.create_reg(Code.POPD_SS, Register.SS))
    a.op(Instruction.create(Code.PUSHFD),
         Instruction.create_reg(Code.POP_R32, Register.EAX))
    a.op(_jmp("ebx"))
    src = _estub(a.build())
    res = simplify(src, STUB_BASE, regs_in=["ebx"], regs_out=["eax", "ecx"])
    assert res.trace.stop is None
    text = _instructions_only(res)
    assert "push ss" not in text and "pop ss" not in text      # pure stack effect
    rep = verify(src, res, runs=25)         # 25 runs: the emulator is reused throughout
    assert rep.ok, str(rep)


def test_a_stub_that_calls_backwards_still_verifies_compacted(src):
    """The fixture's 0x114C000 calls a helper at 0x114BFB0 - *before* its entry.

    A compact blob is laid out in address order (that is what keeps fall-through
    edges adjacent), so without care the blob would begin with the helper; it has to
    start with a jump to the entry instead.
    """
    for mode in ("inplace", "compact"):
        res = simplify(src, 0x114C000, regs_out=["eax"], mode=mode)
        assert res.trace.stop is None
        rep = verify(src, res, runs=30)
        assert rep.ok, f"{mode}: {rep}"


def test_the_fixture_handlers_leave_no_dead_partial_writes(src):
    """No ``mov cl, x`` left standing in front of a folded ``mov cl, y``.

    0x114A8EB had exactly that: the two writes were separated by a *dropped* forward
    jump (and other dead junk), which the fold walk refused to reach across.
    """
    for addr in (0x1141AE7, 0x11435C8, 0x114A3EA, BRANCHY, 0x114A8EB, 0x114C000):
        for mode in ("inplace", "compact"):
            res = simplify(src, addr, regs_in=["ebx"], regs_out=["ebx"], mode=mode)
            if res.trace.stop is not None:
                continue
            assert _repeated_partial_writes(res) == [], f"{addr:#x} {mode}"


def _repeated_partial_writes(res):
    """Consecutive emitted ``mov <8/16 bit view>, <literal>`` of the same view.

    Whenever the fold can collapse such a pair (the second write fully overwrites the
    field the first one set) the first is dead and must be gone; a leftover shows up
    exactly as the ``mov cl, x`` before a folded ``mov cl, y``.
    """
    views = {"al", "cl", "dl", "bl", "ah", "ch", "dh", "bh",
             "ax", "cx", "dx", "bx", "si", "di", "bp", "sp"}
    out: list[str] = []
    prev = None
    for s in res.kept_steps:
        for ins in s.instrs():
            txt = asm.fmt(ins)
            view = None
            if txt.startswith("mov ") and "," in txt:
                dst, _, val = txt[4:].partition(",")
                val = val.strip()
                if dst.strip() in views and (val.startswith("0x") or val.isdigit()):
                    view = dst.strip()
            if view is not None and view == prev:
                out.append(txt)
            prev = view
    return out


def test_a_loop_with_an_exit_inside_its_span_stops():
    """A ret between the loop head and its back edge is not a contiguous region."""
    a = _Asm(STUB_BASE)
    a.label("head")
    a.op(asm.alu("cmp", asm.reg("ecx"), asm.imm(3)))
    a.jcc("e", "tail")
    a.op(Instruction.create(Code.RETND))
    a.label("tail")
    a.jmp("head")
    src = _estub(a.build())
    res = simplify(src, STUB_BASE, regs_in=["ecx", "ebx"], regs_out=["eax"])
    assert res.trace.stop is not None                     # fell back to the linear walk


# ===========================================================================================
# the real branchy handler (0x114756A) and the spin lock (0x1141980)
# ===========================================================================================
BRANCHY = 0x114756A
ARM_STORES = (0x1147653, 0x11477F6)         # [ebx+4] += edx / += 0xa


def _arm_stores_seen(src, runs: int = 60) -> set:
    """Which of the two arm store sites the original executes over ``runs`` seeds."""
    from asmsim.verify import _image
    image = _image(src, 0x1147000, 0x1149000)
    seen: set = set()
    for i in range(runs):
        seed = 0x5EED + i
        emu = Emu(seed)
        for base, data in image.items():
            emu.map_code(base, data)
        emu.set_state(random_regs(random.Random(seed)))
        ip = BRANCHY
        for _ in range(20000):
            if ip in ARM_STORES:
                seen.add(ip)
            if ip == 0x114789F:
                break
            emu.uc.emu_start(ip, 0, count=1)
            ip = emu.uc.reg_read(REGS["eip"])
        emu.close()
    return seen


def test_branchy_handler_inplace(src):
    res = simplify(src, BRANCHY, regs_in=["ebx"], regs_out=["ebx"])
    assert res.trace.stop is None
    assert len(res.exits) == 1
    assert sum(1 for s in res.kept_steps if s.kind == "jcc") == 1
    assert res.stats["ins_kept"] < res.stats["ins_total"] / 2
    assert len(res.code) == res.end - res.start
    assert _repeated_partial_writes(res) == []           # no dead partial writes left
    rep = verify(src, res, runs=40)
    assert rep.ok, str(rep)


def test_branchy_handler_compact_reaches_both_arms(src):
    res = simplify(src, BRANCHY, regs_in=["ebx"], regs_out=["ebx"], mode="compact")
    assert res.trace.stop is None
    assert res.terminator_ip_out is not None
    assert len(res.code) < res.end - res.start            # compact is smaller
    assert _repeated_partial_writes(res) == []
    rep = verify(src, res, runs=60)
    assert rep.ok, str(rep)
    # the branch is real: over these seeds the original walks both arms
    assert _arm_stores_seen(src, runs=60) == set(ARM_STORES)


def test_spin_lock_loop_is_kept_and_analysed_without_keep(src):
    # the lock byte is only seeded so the *emulator* can get past the wait (the
    # original deadlocks on a random byte exactly like the simplified one)
    res = simplify(src, 0x1141980, args=["arg0", "arg1"], regs_out=["eax", "edx"],
                   mem={0x14A0240: 0x41})
    assert res.trace.stop is None
    assert res.trace.regions == [(0x1141987, 0x114199D)]
    assert any(lo == 0x1141987 and hi == 0x114199D for lo, hi, _d, _i in res.trace.keep)
    with pytest.raises(StubError):                       # a region only fits inplace
        simplify(src, 0x1141980, args=["arg0", "arg1"], regs_out=["eax", "edx"],
                 mem={0x14A0240: 0x41}, mode="compact")
    rep = verify(src, res, runs=20)
    assert rep.ok, str(rep)


def test_a_keep_range_cutting_into_a_loop_is_refused(src):
    with pytest.raises(ValueError):
        simplify(src, 0x1141980, args=["arg0", "arg1"], regs_out=["eax", "edx"],
                 keep="0x114198a-0x114199d")      # starts inside the loop region


def test_the_old_keep_workflow_still_verifies(src):
    res = simplify(src, 0x1141980, args=["arg0", "arg1"], regs_out=["eax", "edx"],
                   mem={0x14A0240: 0x41}, keep="0x1141987-0x114199d")
    assert res.trace.stop is None
    rep = verify(src, res, runs=20)
    assert rep.ok, str(rep)


# ===========================================================================================
# the slice invariant over the control-flow graph
# ===========================================================================================
def _reaching_defs_are_kept(res):
    """No kept instruction may read a register whose reaching writer (on any path)
    was dropped - the CFG form of the old linear invariant."""
    trace = res.trace
    out: dict = {}
    by_id = {id(s): s for s in trace.steps}
    allnames = list(REG32) + FLAGS
    for s in trace.order:
        if not s.preds:
            cur = {n: frozenset([None]) for n in allnames}
        else:
            cur = {}
            for p in s.preds:
                for n, v in out[id(p)].items():
                    cur[n] = cur.get(n, frozenset()) | v
        if s.kept:
            for n in sorted(set(s.reads) | set(s.flag_reads)):
                if n in ("esp", "df"):
                    continue
                for w in cur.get(n, frozenset()):
                    assert w is None or by_id[w].kept, \
                        f"{s.ip:#x} reads {n} whose writer was dropped"
        after = dict(cur)
        for n in set(s.writes) | set(s.flag_writes):
            after[n] = frozenset([id(s)])
        out[id(s)] = after


def test_reaching_defs_survive_the_slice(src):
    for res in (
        simplify(src, BRANCHY, regs_in=["ebx"], regs_out=["ebx"]),
        simplify(src, BRANCHY, regs_in=["ebx"], regs_out=["ebx"], mode="compact"),
        simplify(src, 0x1141980, args=["arg0", "arg1"], regs_out=["eax", "edx"],
                 mem={0x14A0240: 0x41}),
        simplify(_diamond_stub(), STUB_BASE, regs_in=["eax", "ebx"], regs_out=["ecx"]),
        simplify(_nested_diamond_stub(), STUB_BASE, regs_in=["eax", "edx", "ebx"],
                 regs_out=["ecx"]),
        simplify(_store_diamond_stub(), STUB_BASE, regs_in=["eax", "ebx"], regs_out=[]),
        simplify(_two_exit_stub(), STUB_BASE, regs_in=["eax", "ebx"], regs_out=["eax"]),
    ):
        _reaching_defs_are_kept(res)
