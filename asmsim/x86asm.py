"""Thin builder layer over iced-x86: pick the right opcode form for an operation.

Operands are tiny tuples:  ``("r", name, bits)`` (``bits`` 8 / 16 / 32, ``"h"``
for the high byte registers), ``("i", value)`` and ``("m", Mem)``.
"""
from __future__ import annotations

from dataclasses import dataclass

from iced_x86 import BlockEncoder, Code, Formatter, FormatterSyntax, Instruction, MemoryOperand, Register

from .terms import MASK, sx

R32 = {n: getattr(Register, n.upper()) for n in ("eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi")}
R16 = {"eax": Register.AX, "ecx": Register.CX, "edx": Register.DX, "ebx": Register.BX,
       "esp": Register.SP, "ebp": Register.BP, "esi": Register.SI, "edi": Register.DI}
R8L = {"eax": Register.AL, "ecx": Register.CL, "edx": Register.DL, "ebx": Register.BL}
R8H = {"eax": Register.AH, "ecx": Register.CH, "edx": Register.DH, "ebx": Register.BH}
BYTE_CAPABLE = ("eax", "ecx", "edx", "ebx")
CCS = ("o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g")
NEG_CC = {"o": "no", "no": "o", "b": "ae", "ae": "b", "e": "ne", "ne": "e", "be": "a", "a": "be",
          "s": "ns", "ns": "s", "p": "np", "np": "p", "l": "ge", "ge": "l", "le": "g", "g": "le"}


@dataclass(frozen=True)
class Mem:
    base: str | None = None
    index: str | None = None
    scale: int = 1
    disp: int = 0
    seg: str | None = None            # "fs" / "gs" / None

    def operand(self) -> MemoryOperand:
        disp = sx(self.disp & 0xFFFFFFFF, 32)
        if self.base is None and self.index is None:
            dsz = 4
        elif disp == 0 and self.base != "ebp":
            dsz = 0
        elif -128 <= disp <= 127:
            dsz = 1
        else:
            dsz = 4
        seg = {"fs": Register.FS, "gs": Register.GS}.get(self.seg, Register.NONE)
        return MemoryOperand(R32[self.base] if self.base else Register.NONE,
                             R32[self.index] if self.index else Register.NONE,
                             self.scale, disp, dsz, False, seg)


def reg(name: str, bits: int = 32):
    return ("r", name, bits)


def imm(v: int):
    return ("i", v)


def mem(m: Mem, bits: int = 32):
    return ("m", m, bits)


def regid(op) -> int:
    _, name, bits = op
    if bits == 32:
        return R32[name]
    if bits == 16:
        return R16[name]
    if bits == 8:
        return R8L[name]
    if bits == "h":
        return R8H[name]
    raise ValueError(bits)


def C(name: str):
    return getattr(Code, name)


def _size(op, other=None) -> int:
    for o in (op, other):
        if o is not None and o[0] in ("r", "m"):
            b = o[2]
            return 8 if b == "h" else b
    raise ValueError("operand size unknown")


def alu(opname: str, dst, src, bits: int | None = None) -> Instruction:
    """mov / add / or / adc / sbb / and / sub / xor / cmp / test"""
    n = bits or _size(dst, src)
    U = opname.upper()
    if src[0] == "i":
        v = sx(src[1] & MASK[n], n)
        if opname == "mov":
            code = C(f"MOV_R{n}_IMM{n}") if dst[0] == "r" else C(f"MOV_RM{n}_IMM{n}")
        elif opname == "test":
            code = C(f"TEST_RM{n}_IMM{n}")
        elif n > 8 and -128 <= v <= 127:
            code = C(f"{U}_RM{n}_IMM8")
        else:
            code = C(f"{U}_RM{n}_IMM{n}")
        if dst[0] == "r":
            return Instruction.create_reg_i32(code, regid(dst), v)
        return Instruction.create_mem_i32(code, dst[1].operand(), v)
    if src[0] == "r":
        code = C(f"{U}_RM{n}_R{n}")
        if dst[0] == "r":
            return Instruction.create_reg_reg(code, regid(dst), regid(src))
        return Instruction.create_mem_reg(code, dst[1].operand(), regid(src))
    assert dst[0] == "r" and src[0] == "m"
    return Instruction.create_reg_mem(C(f"{U}_R{n}_RM{n}"), regid(dst), src[1].operand())


def mov(dst, src, bits=None):
    return alu("mov", dst, src, bits)


def lea(dst_name: str, m: Mem) -> Instruction:
    return Instruction.create_reg_mem(Code.LEA_R32_M, R32[dst_name], m.operand())


def unary(opname: str, dst, bits=None) -> Instruction:
    n = bits or _size(dst)
    code = C(f"{opname.upper()}_RM{n}")
    if dst[0] == "r":
        return Instruction.create_reg(code, regid(dst))
    return Instruction.create_mem(code, dst[1].operand())


def bswap(name: str) -> Instruction:
    return Instruction.create_reg(Code.BSWAP_R32, R32[name])


def shift(opname: str, dst, count, bits=None) -> Instruction:
    """count: ("i", k) or ("cl",)"""
    n = bits or _size(dst)
    U = opname.upper()
    if count[0] == "cl":
        code = C(f"{U}_RM{n}_CL")
        if dst[0] == "r":
            return Instruction.create_reg_reg(code, regid(dst), Register.CL)
        return Instruction.create_mem_reg(code, dst[1].operand(), Register.CL)
    code = C(f"{U}_RM{n}_IMM8")
    if dst[0] == "r":
        return Instruction.create_reg_i32(code, regid(dst), count[1])
    return Instruction.create_mem_i32(code, dst[1].operand(), count[1])


def dshift(opname: str, dst_name: str, src_name: str, count) -> Instruction:
    U = opname.upper()
    if count[0] == "cl":
        return Instruction.create_reg_reg_reg(C(f"{U}_RM32_R32_CL"), R32[dst_name], R32[src_name], Register.CL)
    return Instruction.create_reg_reg_i32(C(f"{U}_RM32_R32_IMM8"), R32[dst_name], R32[src_name], count[1])


def imul(dst_name: str, src, imm_value: int | None = None) -> Instruction:
    if imm_value is not None:
        v = sx(imm_value & 0xFFFFFFFF, 32)
        code = C("IMUL_R32_RM32_IMM8") if -128 <= v <= 127 else C("IMUL_R32_RM32_IMM32")
        if src[0] == "r":
            return Instruction.create_reg_reg_i32(code, R32[dst_name], regid(src), v)
        return Instruction.create_reg_mem_i32(code, R32[dst_name], src[1].operand(), v)
    if src[0] == "r":
        return Instruction.create_reg_reg(Code.IMUL_R32_RM32, R32[dst_name], regid(src))
    return Instruction.create_reg_mem(Code.IMUL_R32_RM32, R32[dst_name], src[1].operand())


def movx(kind: str, dst_name: str, src, dst_bits: int = 32) -> Instruction:
    """movzx / movsx from an 8 or 16 bit register or memory operand"""
    sbits = 8 if (src[0] == "r" and src[2] == "h") else (src[2])
    code = C(f"{kind.upper()}_R{dst_bits}_RM{sbits}")
    if src[0] == "r":
        return Instruction.create_reg_reg(code, R32[dst_name] if dst_bits == 32 else R16[dst_name], regid(src))
    return Instruction.create_reg_mem(code, R32[dst_name] if dst_bits == 32 else R16[dst_name], src[1].operand())


def load(dst, m: Mem, bits: int) -> Instruction:
    return mov(dst, mem(m, bits), bits)


def cmov(cc: str, dst_name: str, src) -> Instruction:
    code = C(f"CMOV{cc.upper()}_R32_RM32")
    if src[0] == "r":
        return Instruction.create_reg_reg(code, R32[dst_name], regid(src))
    return Instruction.create_reg_mem(code, R32[dst_name], src[1].operand())


def setcc(cc: str, dst) -> Instruction:
    return Instruction.create_reg(C(f"SET{cc.upper()}_RM8"), regid(dst))


def bt(dst, count) -> Instruction:
    if count[0] == "i":
        return Instruction.create_reg_i32(Code.BT_RM32_IMM8, regid(dst), count[1])
    return Instruction.create_reg_reg(Code.BT_RM32_R32, regid(dst), regid(count))


def bscan(kind: str, dst_name: str, src) -> Instruction:
    code = C(f"{kind.upper()}_R32_RM32")
    if src[0] == "r":
        return Instruction.create_reg_reg(code, R32[dst_name], regid(src))
    return Instruction.create_reg_mem(code, R32[dst_name], src[1].operand())


def xchg(a: str, b: str) -> Instruction:
    return Instruction.create_reg_reg(Code.XCHG_RM32_R32, R32[a], R32[b])


def push(src) -> Instruction:
    if src[0] == "r":
        return Instruction.create_reg(Code.PUSH_R32, regid(src))
    if src[0] == "i":
        v = sx(src[1] & 0xFFFFFFFF, 32)
        return Instruction.create_i32(Code.PUSHD_IMM8 if -128 <= v <= 127 else Code.PUSHD_IMM32, v)
    return Instruction.create_mem(Code.PUSH_RM32, src[1].operand())


def one(code_name: str) -> Instruction:
    return Instruction.create(C(code_name))


def jcc(cc: str, target: int) -> Instruction:
    return Instruction.create_branch(C(f"J{cc.upper()}_REL32_32"), target)


def jmp(target: int) -> Instruction:
    return Instruction.create_branch(Code.JMP_REL32_32, target)


# ---------------------------------------------------------------------------------------
def encode(instrs: list[Instruction], address: int) -> bytes:
    enc = BlockEncoder(32)
    for i in instrs:
        enc.add(i)
    return bytes(enc.encode(address))


_fmt = Formatter(FormatterSyntax.INTEL)
_fmt.uppercase_hex = False
_fmt.hex_prefix = "0x"
_fmt.hex_suffix = ""
_fmt.space_after_operand_separator = True


def fmt(i: Instruction) -> str:
    return _fmt.format(i)
