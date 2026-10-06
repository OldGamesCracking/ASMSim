"""Unicorn based reference emulator used by the tests.

Memory is mapped lazily: the first touch of any page maps it and fills it with
deterministic pseudo random bytes derived from (seed, page address).  Two runs
with the same seed therefore see exactly the same memory, which makes the
original code and the simplified code directly comparable - even when pointers
are random garbage.

The same goes for the clock: ``rdtsc`` is the one instruction whose result is not
a function of the machine state, so it is hooked and given a deterministic,
monotonic value per run (derived from the seed).  Stubs that read the time stamp
counter - a favourite of obfuscators - stay comparable, and one that *spins* until
the counter moves still terminates.
"""
from __future__ import annotations

import random

from unicorn import (Uc, UcError, UC_ARCH_X86, UC_MODE_32, UC_HOOK_CODE, UC_HOOK_MEM_UNMAPPED,
                     UC_HOOK_MEM_WRITE, UC_PROT_ALL)
from unicorn import x86_const as X

PAGE = 0x1000
RDTSC_OPCODE = b"\x0f\x31"
REGS = {"eax": X.UC_X86_REG_EAX, "ecx": X.UC_X86_REG_ECX, "edx": X.UC_X86_REG_EDX,
        "ebx": X.UC_X86_REG_EBX, "esp": X.UC_X86_REG_ESP, "ebp": X.UC_X86_REG_EBP,
        "esi": X.UC_X86_REG_ESI, "edi": X.UC_X86_REG_EDI, "eflags": X.UC_X86_REG_EFLAGS,
        "eip": X.UC_X86_REG_EIP}
FLAG_BITS = {"cf": 0x1, "pf": 0x4, "af": 0x10, "zf": 0x40, "sf": 0x80, "of": 0x800}
ALL_FLAGS = sum(FLAG_BITS.values())


class MemError(Exception):
    pass


_page_cache: dict = {}


def initial_page(seed: int, page: int) -> bytes:
    """The deterministic initial content of a lazily mapped page."""
    key = (seed, page)
    b = _page_cache.get(key)
    if b is None:
        if len(_page_cache) > 4096:
            _page_cache.clear()
        b = _page_cache[key] = random.Random(f"{seed}:{page:#x}").randbytes(PAGE)
    return b


def initial_byte(seed: int, addr: int) -> int:
    return initial_page(seed, addr & ~(PAGE - 1))[addr & (PAGE - 1)]


class Emu:
    def __init__(self, seed: int, code: dict[int, bytes] | None = None):
        self.uc = Uc(UC_ARCH_X86, UC_MODE_32)
        self.seed = seed
        self.pages: set[int] = set()
        self.code_pages: set[int] = set()
        self.page_kind: dict[int, str] = {}         # "code" (overlaid image) or "data" (lazy fill)
        self._code_images: list[tuple[int, bytes]] = []
        self.write_log: list[tuple[int, int]] = []
        self._filling = False                 # the mapper's own mem_writes are not program writes
        self.tsc = (seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF   # deterministic rdtsc
        self._tsc_pending = False
        self.uc.hook_add(UC_HOOK_MEM_UNMAPPED, lambda *a: Emu._unmapped(self, *a))
        self.uc.hook_add(UC_HOOK_MEM_WRITE, lambda *a: Emu._on_write(self, *a))
        self.uc.hook_add(UC_HOOK_CODE, lambda *a: Emu._on_code(self, *a))
        for base, data in (code or {}).items():
            self.map_code(base, data)

    # -- memory ---------------------------------------------------------------------
    def _fill(self, page: int) -> bytes:
        return initial_page(self.seed, page)

    def close(self) -> None:
        """Break the reference cycles so Unicorn can free its (large) allocations."""
        self.uc = None

    def _write_raw(self, addr: int, data: bytes) -> None:
        self._filling = True
        self.uc.mem_write(addr, data)
        self._filling = False

    def ensure(self, addr: int, size: int = 1) -> None:
        if addr < 0 or addr + size > 1 << 32:
            raise MemError(f"address {addr:#x}+{size} out of range")
        first, last = addr & ~(PAGE - 1), (addr + size - 1) & ~(PAGE - 1)
        for p in range(first, last + 1, PAGE):
            if p not in self.pages:
                self.uc.mem_map(p, PAGE, UC_PROT_ALL)
                self._write_raw(p, self._fill(p))
                self.pages.add(p)
                self.page_kind[p] = "data"

    def map_code(self, base: int, data: bytes) -> None:
        self._code_images.append((base, bytes(data)))
        first, last = base & ~(PAGE - 1), (base + len(data) - 1) & ~(PAGE - 1)
        for p in range(first, last + 1, PAGE):
            if p not in self.pages:
                self.uc.mem_map(p, PAGE, UC_PROT_ALL)
                self.pages.add(p)
                self.code_pages.add(p)
            self.page_kind[p] = "code"
        self._apply_code_images(first, last)

    def _apply_code_images(self, first: int, last: int) -> None:
        for p in range(first, last + 1, PAGE):
            content = bytearray(b"\xcc" * PAGE)
            for base, data in self._code_images:
                lo = max(base, p)
                hi = min(base + len(data), p + PAGE)
                if lo < hi:
                    content[lo - p:hi - p] = data[lo - base:hi - base]
            self._write_raw(p, bytes(content))

    def _on_code(self, uc, address, size, user):
        """Give ``rdtsc`` a deterministic, monotonic value (see the module docstring).

        The code hook runs *before* its instruction, so the value is written just
        before the *next* one (or by :meth:`settle` when the caller looks at the
        state) - writing it here would be overwritten by the real ``rdtsc``.
        """
        if self._tsc_pending:
            self._tsc_pending = False
            self._write_tsc()
        if size == 2 and self.read(address, 2) == RDTSC_OPCODE:
            self.tsc = (self.tsc + 0x40) & 0xFFFFFFFFFFFFFFFF
            self._tsc_pending = True

    def _write_tsc(self) -> None:
        self.uc.reg_write(X.UC_X86_REG_EAX, self.tsc & 0xFFFFFFFF)
        self.uc.reg_write(X.UC_X86_REG_EDX, self.tsc >> 32)

    def settle(self) -> None:
        """Apply a pending ``rdtsc`` result (the run may have stopped right after one)."""
        if self._tsc_pending:
            self._tsc_pending = False
            self._write_tsc()

    def reset(self, seed: int | None = None) -> None:
        """Restore every mapped page to its pristine content for ``seed``.

        Code image pages go back to their 0xCC fill plus the mapped data, lazily
        mapped pages get the deterministic garbage of the (possibly new) seed, and
        the write log is emptied - so one emulator can serve many runs.
        """
        if seed is not None:
            self.seed = seed
        self.tsc = (self.seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        self._tsc_pending = False
        for p in sorted(self.pages):
            if self.page_kind.get(p) == "code":
                continue
            self._write_raw(p, self._fill(p))
        for base, data in self._code_images:
            first, last = base & ~(PAGE - 1), (base + len(data) - 1) & ~(PAGE - 1)
            self._apply_code_images(first, last)
        self.write_log.clear()

    def _unmapped(self, uc, access, address, size, value, user):
        try:
            self.ensure(address, max(size, 1))
        except (UcError, MemError):
            return False
        return True

    def _on_write(self, uc, access, address, size, value, user):
        if not self._filling:
            self.write_log.append((address, size))

    def write_image(self, addr: int, data: bytes) -> None:
        """Patch mapped memory (e.g. put the simplified code over the original)."""
        self._write_raw(addr, bytes(data))

    def read(self, addr: int, size: int) -> bytes:
        self.ensure(addr, size)
        return bytes(self.uc.mem_read(addr, size))

    def read_int(self, addr: int, size: int) -> int:
        return int.from_bytes(self.read(addr, size), "little")

    # -- execution -------------------------------------------------------------------
    def set_state(self, regs: dict[str, int]) -> None:
        for n, v in regs.items():
            self.uc.reg_write(REGS[n], v & 0xFFFFFFFF)

    def get_state(self) -> dict[str, int]:
        self.settle()
        return {n: self.uc.reg_read(r) for n, r in REGS.items()}

    def run(self, start: int, until: int, max_insns: int = 200000, count: int = 0) -> None:
        self.write_log.clear()
        self.ensure(start, 1)
        self.uc.emu_start(start, until, count=count or max_insns)

    def written_bytes(self) -> dict[int, int]:
        out = {}
        for a, s in self.write_log:
            for i in range(s):
                out[(a + i) & 0xFFFFFFFF] = self.read(a + i, 1)[0]
        return out


def random_regs(rng: random.Random, esp_base: int = 0x00300000) -> dict[str, int]:
    r = {n: rng.getrandbits(32) for n in ("eax", "ecx", "edx", "ebx", "ebp", "esi", "edi")}
    r["esp"] = esp_base + (rng.randrange(0x4000, 0x8000) & ~3)
    fl = 0x2
    for b in FLAG_BITS.values():
        if rng.random() < 0.5:
            fl |= b
    r["eflags"] = fl
    return r
