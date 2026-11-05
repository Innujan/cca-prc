import galois
import numpy as np
from pseudorandom_primitives import PRG
import secrets
import os
import json
import math

_POLY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lowweight_irreducible.json")
_FIELD_CACHE = {}
 
 
# ---------------------------------------------------------------------- GF(2)[x] su interi
def clmul(a, b):
    if a.bit_length() < b.bit_length():
        a, b = b, a
    r = 0
    while b:
        lsb = b & -b
        r ^= a * lsb                    # a * 2^k = a << k
        b ^= lsb
    return r
 
 
def _spread_byte(x):
    return sum(((x >> i) & 1) << (2 * i) for i in range(8))
 
 
_SPREAD = [_spread_byte(x).to_bytes(2, "little") for x in range(256)]
 
 
def sqr(a):
    if a == 0:
        return 0
    raw = a.to_bytes((a.bit_length() + 7) // 8, "little")
    return int.from_bytes(b"".join(_SPREAD[x] for x in raw), "little")
 
 
def pmod(a, f):
    df = f.bit_length() - 1
    da = a.bit_length() - 1
    while da >= df:
        a ^= f << (da - df)
        da = a.bit_length() - 1
    return a
 
 
def pgcd(a, b):
    while b:
        a, b = b, pmod(a, b)
    return a
 
 
def _reduce_sparse(x, l, low_exps, mask):
    while x >> l:
        hi = x >> l
        x &= mask
        for e in low_exps:
            x ^= hi << e
    return x
 
 
def is_irreducible(exps):
    l = exps[0]
    f = sum(1 << e for e in exps)
    low = list(exps[1:])
    mask = (1 << l) - 1
    u = 2                                       # x
    for _ in range(l // 2):
        u = _reduce_sparse(sqr(u), l, low, mask)
        if pgcd(f, u ^ 2) != 1:
            return False
    return True
 
 
def find_low_weight_irreducible(l):
    if l < 2:
        raise ValueError("grado troppo piccolo")
    for a in range(1, l // 2 + 1):
        if is_irreducible((l, a, 0)):
            return (l, a, 0)
    for a in range(3, l):
        for b in range(2, a):
            for c in range(1, b):
                if is_irreducible((l, a, b, c, 0)):
                    return (l, a, b, c, 0)
    raise ValueError(f"nessun polinomio di peso <= 5 di grado {l}")
 
 
def _load_cache():
    try:
        with open(_POLY_FILE) as f:
            return {int(k): tuple(v) for k, v in json.load(f).items()}
    except (OSError, ValueError):
        return {}
 
 
def irreducible_poly(l):
    cache = _load_cache()
    if l not in cache:
        cache[l] = find_low_weight_irreducible(l)
        try:
            with open(_POLY_FILE, "w") as f:
                json.dump({str(k): list(v) for k, v in sorted(cache.items())}, f)
        except OSError:
            pass
    return cache[l]
 
 
class GF2m:
    def __init__(self, l, exps=None):
        self.l = l
        self.exps = tuple(exps) if exps is not None else irreducible_poly(l)
        if self.exps[0] != l or any(e >= l for e in self.exps[1:]):
            raise ValueError("polinomio non valido per il grado l")
        self.low = self.exps[1:]
        self.mask = (1 << l) - 1
        self.order = 1 << l
 
    def __call__(self, x):                      # conversione, come galois.GF(...)(x)
        x = int(x)
        if x < 0:
            raise ValueError("elemento negativo")
        return _reduce_sparse(x, self.l, self.low, self.mask)
 
    def mul(self, a, b):
        return _reduce_sparse(clmul(a, b), self.l, self.low, self.mask)
 
    def random(self):
        return secrets.randbits(self.l)
 
    def __repr__(self):
        terms = " + ".join(f"x^{e}" if e > 1 else ("x" if e == 1 else "1") for e in self.exps)
        return f"GF(2^{self.l}) mod {terms}"
 
 
def get_field(l):
    if l not in _FIELD_CACHE:
        _FIELD_CACHE[l] = GF2m(l)
    return _FIELD_CACHE[l]
 
class Com:
    def __init__(self, l_bytes=16, tag_bits=256):
        self.l_bytes = l_bytes
        self.l_bits = l_bytes*8
        self.tag_bits = tag_bits
        self.l_out = self.tag_bits+3*self.l_bits
        self.field = get_field(self.l_out)
        self.prg = PRG()
        self._btag = {}

    def _tag_to_field(self, tag):
        tag_int = int.from_bytes(tag, byteorder='big')
        if tag_int >= (1 << self.tag_bits):
            raise ValueError(f"tag larger than {self.tag_bits} bits")
        return tag_int

    def _B_tag(self, B, tag):
        key = (B, tag)
        if key not in self._btag:
            if len(self._btag) > 64:
                self._btag.clear()
            self._btag[key] = self.field.mul(B, self._tag_to_field(tag))
        return self._btag[key]
    
    def setup(self, n):
        B = self.field(secrets.randbits(self.l_out))
        A = [self.field(secrets.randbits(self.l_out)) for _ in range(n)]
        return (B, A, n, self.field)
    
    def commit(self, pp, S, tag):
        B, A, n, field = pp
        if len(S) != n: raise ValueError(f"length mismatch len(S) = {len(S)} != {n}")

        tag_val = self._tag_to_field(tag)
        comms = []
        ops = []
        Btag = self._B_tag(B, tag)
        for i in range(n):
            op = secrets.token_bytes(self.l_bytes)
            prg_op = self.prg.generate(op, self.l_out, field)
            com = prg_op
            if (S[i] == 1):
                com ^= A[i] ^ Btag
            comms.append(com)
            ops.append(op)
        return comms, ops

    def open_com(self, pp, i, b, com, op, tag):
        B, A, n, field = pp
        if not 0 <= 1 < n or b not in (0, 1):
            return False
        expect = self.prg.generate(op, self.l_out, field)
        if b == 1:
            expect ^= A[i] ^ self._B_tag(B, tag)
        return int(com) == expect

    def equiv(self, n, tag):
        tag_val = self._tag_to_field(tag)
        B = self.field.random()
        Bt = self.field.mul(B, tag_val)

        A = []
        comms = []
        ops0 = []
        ops1 = []

        for _ in range(n):
            v = secrets.token_bytes(self.l_bytes)
            w = secrets.token_bytes(self.l_bytes)
            prg_v = self.prg.generate(v, self.l_out, self.field)
            prg_w = self.prg.generate(w, self.l_out, self.field)

            comms.append(prg_v)
            A.append(prg_v ^ prg_w ^ Bt)
            ops0.append(v)
            ops1.append(w)
        
        return (B, A, n, self.field), comms, ops0, ops1

if __name__ == "__main__":
    com = Com(l_bytes=16, tag_bits=16)
    n = 4
    tag = secrets.token_bytes(2)
    S = [0, 1, 1, 0]

    pp = com.setup(n)
    comms, ops = com.commit(pp, S, tag)

    for i in range(n):
        assert com.open_com(pp, i, S[i], comms[i], ops[i], tag) is True
        assert com.open_com(pp, i, 1-S[i], comms[i], ops[i], tag) is False
    print("correctness test passed")

    pp_equiv, comms_equiv, ops0, ops1 = com.equiv(n, tag)

    for i in range(n):
        assert com.open_com(pp_equiv, i, 0, comms_equiv[i], ops0[i], tag) is True
        assert com.open_com(pp_equiv, i, 1, comms_equiv[i], ops1[i], tag) is True
    print("equivocality test passed")

    try:
        com.commit(pp, S, secrets.token_bytes(3))
        raise AssertionError("tag troppo lungo accettato")
    except ValueError:
        print("tag length check passed")