import galois
import numpy as np
from pseudorandom_primitives import PRG
import secrets
import os
import json
import math
from fast_GF import get_field
 
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
        if not 0 <= i < n or b not in (0, 1):
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