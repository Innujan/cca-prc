import numpy as np
import math

class HPRG:
    def __init__(self, lam=32, l=128, d=16, esp=0.5, k=None, c_ecc=1.5, eps=None, n=None, seed=None):
        self.lam = lam
        self.l = l
        self.d = d
        self.esp = esp
        self.k = k
        self.c_ecc = c_ecc
        self.n = n
        self.eps = (2.0**(-lam)) if eps is None else eps
        self.rng = np.random.default_rng(seed)

    def setup(self):
        lam = self.lam
        l = self.l
        d = self.d
        esp = self.esp
        c_ecc = self.c_ecc

        p_exp = int(math.ceil(d ** esp))
        p = 1 << p_exp
        s = 3.2
        log2_inv_eps = math.log2(1.0 / self.eps) if self.eps > 0 else 2 * lam
        min_entropy = d * p_exp + 2 * log2_inv_eps
        self.k = int(math.ceil(min_entropy))+1 if self.k is None else self.k
        if self.k < min_entropy:
            raise ValueError("entropic condition unsatisfied for LHL")
        n = self.n if self.n is not None else int(math.ceil(self.k*self.c_ecc))
        if n < self.k:
            raise ValueError("n must be >= k")
        self.n = n

        dtype = np.int16 if p_exp <= 14 else np.int32
        rng = self.rng

        A = rng.integers(low=0, high=p, size=(n, 2, d), dtype=np.int16)
        v0 = rng.integers(low=0, high=p, size=(l, d), dtype=np.int16)

        H0 = np.einsum('kbd,jd->jbk', A, v0) % p
        e0 = np.rint(rng.normal(loc=0.0, scale=s, size=(l, 2, n))).astype(np.int16)
        H0 = ((H0 + e0) % p).astype(dtype)

        H = np.empty((n, l, 2, 2, n), dtype=dtype)

        for i in range(n):
            Vi = rng.integers(low=0, high=p, size=(l, 2, d), dtype=np.int16)
            Hi = np.einsum('kcd,jbd->jbck', A, Vi) % p
            ei = np.rint(rng.normal(loc=0.0, scale=s, size=(l, 2, 2, n))).astype(np.int16)
            H[i] = ((Hi + ei) % p).astype(dtype)
            H[i, :, 0, 0, i] = -1
            H[i, :, 1, 1, i] = -1

        return (H0, H, p, p_exp)

    def eval_seed(self, pp, s, i):
        H0, H, p, p_exp = pp
        l = H0.shape[0]
        n = H.shape[0]
        Z = []
        if not 0 <= i <= n:
            raise IndexError(f"index out of range")
        ks = np.arange(n)
        if i != 0:
            ib = i - 1
            vals = H[ib][:, s[ib]][:, s, ks].astype(np.int16)
            mask = ks != ib
            ret = vals[:, mask].sum(axis=1)
        else:
            ret = H0[:, s, ks].astype(np.int16).sum(axis=1)
        ret %= p
        return ((ret >> (p_exp-1)) & 1).astype(np.uint8)
    
    def eval_all(self, pp, s):
        n = pp[1].shape[0]
        return np.stack([self.eval_seed(pp, s, i) for i in range(n+1)])

if __name__ == "__main__":
    hp = HPRG(lam=8, l=16, d=4, esp=0.5)
    pp = hp.setup()
    n = pp[1].shape[0]
    s = np.random.randint(0, 2, size=n)
    y0 = hp.eval_seed(pp, s, 0)
    assert len(y0) == hp.l
    for i in range(1, n + 1):  # incluso i = n
        z = hp.eval_seed(pp, s, i)
        assert len(z) == hp.l
        assert (z == hp.eval_seed(pp, s, i)).all()
    try:
        hp.eval_seed(pp, s, n + 1)
        raise AssertionError("i = n+1 dovrebbe essere rifiutato")
    except IndexError:
        pass
    print("hprg eval test passed (n =", n, ")")