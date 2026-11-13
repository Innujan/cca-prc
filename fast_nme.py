import math
import random
import numpy as np
import hashlib
import galois
try:  # Sage serve solo per mode="HERMITIAN"
    from sage.all import GF as SageGF, Integer, AffineSpace, Curve, vector
    import sage.coding.codes_catalog as codes
    HAVE_SAGE = True
except ImportError:
    HAVE_SAGE = False
from itertools import combinations

GF2 = galois.GF(2)

class BaseECC:
    def __init__(self, n, d):
        code = galois.BCH(n, d=d)
        self.G = code.G
        F = code.extension_field
        self.n = code.n
        self.k = code.k
        self.t = code.t
        q = F.order-1
        a = code.alpha
        self.exp = np.array([int(a**i) for i in range(q)]*2, dtype=np.int64)
        self.log = np.zeros(q+1, dtype=np.int64)
        self.log[self.exp[:q]] = np.arange(q)
        self.q = q
        self.s = F.degree
        deg = (self.n-1-np.arange(self.n))[:, None]
        js = (code.c + np.arange(2*self.t))[None, :]
        vals = self.exp[(deg*js)%q]
        bits = (vals[:, :, None] >> np.arange(self.s)) & 1
        self.Msyn = bits.reshape(self.n, -1).astype(np.float32)
        self.pw = 1 << np.arange(self.s)
        self.c0 = code.c

    def _mul(self, x, y):
        out = self.exp[self.log[x] + self.log[y]]
        return np.where((x == 0) | (y == 0), 0, out)

    def _inv(self, x):
        return self.exp[(self.q-self.log[x]) % self.q]

    def decode(self, y):
        y = np.asarray(y, dtype=np.uint8)
        n, t, q = y.shape[0], self.t, self.q
        s = (((y.astype(np.float32) @ self.Msyn).astype(np.int64) & 1).reshape(n, 2*t, self.s)*self.pw).sum(axis=2)

        l = np.zeros(n, dtype=np.int64)
        lam = np.zeros((n, 2*t+1), dtype=np.int64)
        lam[:, 0] = 1
        B = lam.copy()
        b = np.ones(n, dtype=np.int64)
        mshift = np.ones(n, dtype=np.int64)
        rows = np.arange(n)[:, None]
        for r in range(2*t):
            i = r - np.arange(2*t+1)
            sr = np.where(i >= 0, s[:, np.clip(i, 0, None)], 0)
            d = np.bitwise_xor.reduce(self._mul(lam, sr), axis=1)
            nz = d != 0
            coef = self._mul(d, self._inv(np.where(b == 0, 1, b)))
            sh = np.arange(2 * t + 1)[None, :] - mshift[:, None]
            Bs = np.where(sh >= 0, B[rows, np.clip(sh, 0, None)], 0)
            Tn = lam ^ self._mul(coef[:, None], Bs)
            grow = nz & (2 * l <= r)
            B = np.where(grow[:, None], lam, B)
            b = np.where(grow, d, b)
            l = np.where(grow, r + 1 - l, l)
            mshift = np.where(grow, 1, mshift + 1)
            lam = np.where(nz[:, None], Tn, lam)
 
        lam = lam[:, :t + 1]
        p = np.arange(self.n)
        ev = np.zeros((n, self.n), dtype=np.int64)
        for i in range(t + 1):
            term = self.exp[(-i * p) % q][None, :]
            ev ^= self._mul(lam[:, i:i + 1], term)
        roots = ev == 0
        nroots = roots.sum(axis=1)
        deg = np.where(lam != 0, np.arange(t + 1)[None, :], 0).max(axis=1)
        ok = (l <= t) & (deg == l) & (nroots == l)
        E = roots[:, ::-1]
        C = y ^ E.astype(np.uint8)
        nerr = np.where(ok, l, -1)
        return C[:, :self.k], nerr
        

class NoisyMcEliece:
    def __init__(self, mbits, n=65, m=255, d=37, rho=1/65, epsilon=3/255, noise=0.01):
        self.n = n
        self.m = m
        self.d = d
        self.rho = rho
        self.epsilon = epsilon
        self.noise = noise
        self.rng = np.random.default_rng()
        self.w = math.floor(rho*n)
        self._sizes = [math.comb(n, k) for k in range(self.w+1)]
        self.bpb = int(math.floor(math.log2(sum(self._sizes))))
        self.code = BaseECC(self.m, d=self.d)
        self.m_blk = self.code.n
        self.t = self.code.t
        self.G = np.asarray(self.code.G.T, dtype=np.uint8)
        self.k = self.G.shape[1]
        self.e_max = math.ceil(2*self.noise*self.m_blk)
        self.eps_w = math.floor(epsilon*self.m_blk)
        self.alpha = self.t - self.eps_w - 2*self.w*self.e_max
        if self.alpha < 1:
            raise ValueError(f"alpha = {self.alpha} < 1: t={self.t}, eps*m={self.eps_w}, "
                             f"2w*e_max={2 * self.w * self.e_max}")
        self.mbits = mbits
        self.nblocks = math.ceil(mbits / self.bpb)
        self.cwbits = self.nblocks*self.m_blk
        self._p_s = np.array(self._sizes, dtype=float) / sum(self._sizes)
        sz_e = [math.comb(self.m_blk, k) for k in range(self.eps_w+1)]
        self._p_e = np.array(sz_e, dtype=float) / sum(sz_e)

    def get_full_column_rank(self, n, k):
        while True:
            T = GF2.Random((k, 2 * n), seed=int(self.rng.integers(1 << 31)))
            if np.linalg.matrix_rank(T) == 2 * n:
                return T

    def find_row_subset(self, T):
        k, cols = T.shape
        while True:
            R = np.sort(np.array(random.sample(range(k), cols))) #retrieve random 2n rows indices
            try: #probability of finding a nxn invertible random matrix = ~0.29
                # probability of T[R] being invertible = wt(v)/k
                Tinv = np.linalg.inv(T[R]) #tries to invert T on the indices of R
                return R, Tinv
            except np.linalg.LinAlgError:
                continue

    def get_sample(self, radius, n, rng=None):
        rng = self.rng if rng is None else rng
        r = math.floor(radius*n)
        sphere_sizes = [math.comb(n, k) for k in range(r+1)]
        total = sum(sphere_sizes)
        k = int(rng.choice(r+1, p=[sz/total for sz in sphere_sizes]))
        result = np.zeros(n, dtype=int)
        if k:
            result[rng.choice(n, size=k, replace=False)] = 1
        return result
    
    def f2_to_f16(self, msg, Fq):
        (a, ) = Fq._first_ngens(1)
        msg16 = []
        for i in range(len(msg)//4):
            w = msg[4*i:4*i+4]
            p16 = Fq(0)
            for j in range(len(w)):
                if w[j] == 1:
                    p16 += a**Integer(j)
            msg16.append(p16)
        return msg16

    def f16_to_f2(self, msg):
        msg2 = []
        for p in msg:
            bits = p.polynomial().list()
            if len(bits) < 4:
                bits.extend([0, ]*(4-len(bits)))
            word = [int(i) for i in bits]
            msg2.extend(word)
        return np.asarray(msg2, dtype=int)
    
    def get_hamming_sphere(self, rng, N, size, p):
        weights = rng.choice(len(p), size=N, p=p)
        order = np.argsort(rng.random((N, size)), axis=1)
        out = np.zeros((N, size), dtype=np.uint8)
        mask = np.arange(size)[None, :] < weights[:, None]
        rows = np.repeat(np.arange(N), weights)
        out[rows, order[mask]] = 1
        return out
    
    def unrank_integer(self, N):
        k = 0
        while N >= self._sizes[k]:
            N -= self._sizes[k]
            k+= 1
        v = np.zeros(self.n, dtype=int)
        x = self.n
        for j in range(k, 0, -1):
            x -= 1
            while math.comb(x, j) > N:
                x -= 1
            v[x] = 1
            N -= math.comb(x, j)
        return v

    def rank_vector(self, v):
        pos = sorted(np.flatnonzero(np.asarray(v, dtype=int)))
        k = len(pos)
        return sum(self._sizes[:k]) + sum(math.comb(int(p), j+1) for j, p in enumerate(pos))

    @staticmethod
    def _bits_to_int(bits):
        return int("".join(map(str, bits)), 2) if len(bits) else 0
    
    def _int_to_bits(self, N):
        return np.array(list(map(int, format(N, f"0{self.bpb}b"))), dtype=np.uint8)


    def KeyGen(self):
        m, n, k, rng = self.m, self.n, self.k, self.rng
        perm = rng.permutation(m)
        T = self.get_full_column_rank(n, k)
        R, Tinv = self.find_row_subset(T)
        E = (rng.random((m, 2*n))<self.noise).astype(np.uint8)
        while True:
            bad = np.flatnonzero(E.sum(axis=0)>self.e_max) # check if matrix E has more than e_max ones in any column
            if bad.size == 0:
                break
            E[:, bad] = (rng.random((m, bad.size)) < self.noise).astype(np.uint8) # regenerate the columns of E that have too many ones
        T, Tinv = np.asarray(T, dtype=np.uint8), np.asarray(Tinv, dtype=np.uint8)
        A = (((self.G.astype(np.int32) @ T) & 1)[perm] ^ E).astype(np.uint8)
        pk = A
        sk = dict(perm=perm, inv=np.argsort(perm), T=T, R=R, Tinv=Tinv)
        return pk, sk

    def Encode(self, pk, msg, seed=None):
        if seed is None:
            rng = self.rng
        else:
            rng = np.random.default_rng(int.from_bytes(hashlib.shake_256(seed).digest(32), byteorder='big'))
        msg = np.asarray(msg, dtype=np.uint8)
        if msg.shape != (self.mbits,):
            raise ValueError(f"length mismatch: {msg.shape} != {self.mbits}")
        N = self.nblocks
        padded = np.zeros(N*self.bpb, dtype=np.uint8)
        padded[:self.mbits] = msg
        s = self.get_hamming_sphere(rng, N, self.n, self._p_s)
        e = self.get_hamming_sphere(rng, N, self.m_blk, self._p_e)
        msg = np.stack([self.unrank_integer(self._bits_to_int(padded[j*self.bpb:(j+1)*self.bpb])) for j in range(N)]).astype(np.uint8)
        x = np.hstack([msg, s]).astype(np.float32) 
        c = ((x @ np.asarray(pk, dtype=np.float32).T).astype(np.int32) & 1).astype(np.uint8) ^ e
        return c.reshape(-1)
                

    def Decode(self, sk, c):
        c = np.asarray(c, dtype=np.uint8)
        if c.shape != (self.cwbits,): return None
        Y = c.reshape(self.nblocks, self.m_blk)[:, sk["inv"]]
        y, nerr = self.code.decode(Y)
        if np.any(nerr < 0):
            return None
        mm = lambda a, b: (a.astype(np.float32) @ b.astype(np.float32)).astype(np.int32) & 1
        if np.any((mm(y, self.G.T) ^ Y).sum(axis=1) > self.t):
            return None
        x = mm(y[:, np.asarray(sk["R"])], sk["Tinv"].T)
        if np.any(mm(x, sk["T"].T) != y):
            return None
        msg, s = x[:, :self.n], x[:, self.n:]
        if (s.sum(axis=1) > self.w).any() or (msg.sum(axis=1) > self.w).any():
            return None
        bits = []
        for m in msg:
            N = self.rank_vector(m)
            if N >= 1 << self.bpb:
                return None
            bits.append(self._int_to_bits(N))
        bits = np.concatenate(bits)
        if bits[self.mbits:].any():
            return None
        return bits[:self.mbits]

def test(trials=5):
    import os
    p = dict(mbits=300, n=65, m=255, d=37, rho=1/65, epsilon=3/255, noise=0.01)
    nme = NoisyMcEliece(p["mbits"], p["n"], p["m"], p["d"], p["rho"], p["epsilon"], p["noise"])
    print(f"w={nme.w}  bpb={nme.bpb}  blocchi={nme.nblocks}  cw_bits={nme.cwbits}  "
          f"t={nme.t}  e_max={nme.e_max}  alpha_bits={nme.alpha}")
    tot = sum(nme._sizes)
    for N in list(range((min(tot, 200)))) + [tot-1]:
        assert nme.rank_vector(nme.unrank_integer(N)) == N
    pk, sk = nme.KeyGen()
    rng = np.random.default_rng(1)
    for t in range(trials):
        msg = rng.integers(0, 2, 300).astype(np.uint8)
        seed = os.urandom(32)
        cw = nme.Encode(pk, msg, seed)
        assert np.array_equal(cw, nme.Encode(pk, msg, seed))
        out = nme.Decode(sk, cw)
        assert out is not None and np.array_equal(out, msg)
        bad = cw.copy()
        bad[:nme.alpha] ^= 1
        bad_out = nme.Decode(sk, bad)
        assert np.array_equal(bad_out, msg)
    rnd = rng.integers(0, 2, nme.cwbits).astype(np.uint8)
    print(nme.Decode(sk, rnd) is None)

if __name__ == "__main__":
    test()