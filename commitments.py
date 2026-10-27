import galois
import numpy as np
from pseudorandom_primitives import PRG
import secrets
import math

_FIELD_CACHE = {}


def _get_field(l_out):
    if l_out not in _FIELD_CACHE:
        poly = galois.irreducible_poly(2, l_out, method="min")
        _FIELD_CACHE[l_out] = galois.GF(2**l_out, irreducible_poly=poly)
    return _FIELD_CACHE[l_out]


class Com:
    def __init__(self, l_bytes=16, tag_bits=256):
        self.l_bytes = l_bytes
        self.l_bits = l_bytes*8
        self.tag_bits = tag_bits
        self.l_out = self.tag_bits+3*self.l_bits
        self.field = _get_field(self.l_out)
        self.prg = PRG()

    def _tag_to_field(self, tag):
        tag_int = int.from_bytes(tag, byteorder='big')
        if tag_int >= (1 << self.tag_bits):
            raise ValueError(f"tag larger than {self.tag_bits} bits")
        return self.field(tag_int)
    
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
        for i in range(n):
            op = secrets.token_bytes(self.l_bytes)
            prg_op = self.prg.generate(op, self.l_out, self.field)
            com = prg_op
            if (S[i] == 1):
                com += A[i] + (B*tag_val)
            comms.append(com)
            ops.append(op)
        return comms, ops

    def open_com(self, pp, i, b, com, op, tag):
        B, A, n, field = pp
        tag_val = self._tag_to_field(tag)
        prg_op = self.prg.generate(op, self.l_out, self.field)
        if (b == 0 and com == prg_op) or (b == 1 and com == prg_op + A[i] + (B*tag_val)):
            return True
        else:
            return False

    def equiv(self, n, tag):
        tag_val = self._tag_to_field(tag)
        B = self.field(secrets.randbits(self.l_out))

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
            A.append(prg_v-prg_w-(B*tag_val))
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