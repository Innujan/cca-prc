import galois
import numpy as np
from pseudorandom_primitives import PRG
import secrets
import math

class Com:
    @staticmethod
    def emb(t_bytes, t_bits, field):
        t_int = int.from_bytes(t_bytes, byteorder='big')
        if t_int >= (1 << t_bits):
            raise ValueError(f"tag larger than {t_bits} bits")
        return field(t_int)

    @staticmethod
    def sample_canonical_polynomial(B, field):
        coeffs = field.Random(B)
        return galois.Poly(coeffs, field=field)

    def __init__(self, lam, N, B, t, override):
        self.lam = lam
        self.N = N
        self.B = B
        self.t = t
        log2N = math.ceil(math.log2(N)) if N > 1 else 1
        self.l = 2*t + (B+1)*log2N+lam*(B+1)+lam if override is None else override
        poly = galois.irreducible_poly(2, self.l, method="min")
        self.GF2l = galois.GF(2**self.l, irreducible_poly=poly)
        self.prg = PRG()
    
    def setup(self):
        A = self.GF2l.Random(self.N)
        D = self.GF2l.Random(self.N)
        return (self.l, A, D)
    
    def commit(self, pp, S, tag):
        l, A, D = pp
        emb_tag = self.emb(tag, self.t, self.GF2l)

        proofs = {}
        xs = []
        ys = []

        for i in S:
            s_i = secrets.token_bytes(math.ceil(self.lam / 8))
            proofs[i] = s_i
            y_i = self.prg.generate(s_i, l, self.GF2l) + A[i] + D[i]*emb_tag
            xs.append(self.GF2l(i+1))
            ys.append(y_i)

        p = galois.lagrange_poly(self.GF2l(xs), self.GF2l(ys))
        return p, proofs

    def verify(self, pp, p, i, sigma_i, tag):
        l, A, D = pp
        emb_tag = self.emb(tag, self.t, self.GF2l)
        prval = self.prg.generate(sigma_i, l, self.GF2l)
        check = prval + A[i] + D[i]*emb_tag

        return bool(p(self.GF2l(i+1)) == check)
        
    def alt_setup(self, tag):
        emb_tag = self.emb(tag, self.t, self.GF2l)
        s = [secrets.token_bytes(math.ceil(self.lam / 8)) for _ in range(self.N)]
        D = self.GF2l.Random(self.N)
        p = self.sample_canonical_polynomial(self.B, self.GF2l)
        A = [p(self.GF2l(i+1))-self.prg.generate(s[i], self.l, self.GF2l)-D[i]*emb_tag for i in range(self.N)]
        pp = (self.l, A, D)
        proofs = {i: s[i] for i in range(self.N)}
        return pp, p, proofs