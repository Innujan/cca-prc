import numpy as np
import math

class HPRG:
    def __init__(self, lam=32, l=128, d=16, esp=0.5, k=None, c_ecc=1.5, eps=None):
        self.lam = lam
        self.l = l
        self.d = d
        self.esp = esp
        self.k = lam if k is None else k
        self.c_ecc = c_ecc
        self.eps = (2.0**(-lam)) if eps is None else eps

    def setup(self):
        lam = self.lam
        l = self.l
        d = self.d
        ep = self.ep
        k = self.k
        c_ecc = self.c_ecc

        p_exp = int(math.ceil(d ** esp))
        p = 1 << p_exp
        s = 3.2
        log2_inv_eps = math.log2(1.0 / self.eps) if self.eps > 0 else 2 * lam
        min_entropy = d * p_exp + 2 * log2_inv_eps
        if k < min_entropy:
            raise ValueError("entropic condition unsatisfied for LHL")
        n = int(math.ceil(k*c_ecc))

        A = np.random.randint(low=0, high=p, size=(n, 2, d), dtype=np.int16)
        v0 = np.random.randint(low=0, high=p, size=(l, d), dtype=np.int16) 
        V = np.random.randint(low=0, high=p, size=(n, l, 2, d), dtype=np.int16)

        H0 = np.einsum('kbd,jd->jbk', A, v0) % p
        e0 = np.rint(np.random.normal(loc=0.0, scale=s, size=(l, 2, n))).astype(np.int16)
        H0 = (H0 + e0) % p

        H = np.einsum('kcd,ijbd->ijbck', A, V) % p
        e = np.rint(np.random.normal(loc=0.0, scale=s, size=(n, l, 2, 2, n))).astype(np.int16)
        H = (H + e) % p

        for i in range(n):
            H[i, :, 0, 0, i] = -1
            H[i, :, 1, 1, i] = -1

        return (H0, H, p, p_exp)

    def eval_seed(self, pp, s, i):
        H0, H, p, p_exp = pp
        l = H0.shape[0]
        n = H.shape[0]
        Z = []
        if not 0 <= i <= n:
            raise IndexError(f"i = {i} fuori da [0, {n}]")
        if i != 0:
            # convenzione del paper: i in [n], quindi blocco i-1 e bit s[i-1] (indici 0-based)
            ib = i - 1
            for j in range(l):
                h_sum = 0
                for k in range(n):
                    if k == ib:
                        continue
                    h_sum = (h_sum + int(H[ib, j, s[ib], s[k], k])) % p
                zj = (h_sum >> (p_exp - 1)) & 1
                Z.append(zj)
        else:
            for j in range(l):
                h_sum = 0
                for k in range(n):
                    h_sum = (h_sum + int(H0[j, s[k], k])) % p
                zj = (h_sum >> (p_exp - 1)) & 1
                Z.append(zj)
        return Z

if __name__ == "__main__":
    hp = HPRG(lam=8, l=16, d=4, ep=0.5)
    pp = hp.setup()
    n = pp[1].shape[0]
    s = np.random.randint(0, 2, size=n)
    y0 = hp.eval_seed(pp, s, 0)
    assert len(y0) == hp.l
    for i in range(1, n + 1):  # incluso i = n
        z = hp.eval_seed(pp, s, i)
        assert len(z) == hp.l
        assert z == hp.eval_seed(pp, s, i)  # deterministico
    try:
        hp.eval_seed(pp, s, n + 1)
        raise AssertionError("i = n+1 dovrebbe essere rifiutato")
    except IndexError:
        pass
    print("hprg eval test passed (n =", n, ")")