import numpy as np
import math

class HPRG:
    def __init__(self, lam=32, l=128, d=16, ep=0.5):
        self.lam = lam
        self.l = l
        self.d = d
        self.ep = ep

    def setup(self):
        lam = self.lam
        l = self.l
        d = self.d
        ep = self.ep

        p_exp = int(math.ceil(d ** ep))
        p = 1 << p_exp
        s = 3.2
        n = d * p_exp + 2*lam

        A = np.random.randint(low=0, high=p, size=(n, 2, d), dtype=np.int64)
        v0 = np.random.randint(low=0, high=p, size=(l, d), dtype=np.int64) 
        V = np.random.randint(low=0, high=p, size=(n, l, 2, d), dtype=np.int64)

        H0 = np.einsum('kbd,jd->jbk', A, v0) % p
        e0 = np.rint(np.random.normal(loc=0.0, scale=s, size=(l, 2, n))).astype(np.int64)
        H0 = (H0 + e0) % p

        H = np.einsum('kcd,ijbd->ijbck', A, V) % p
        e = np.rint(np.random.normal(loc=0.0, scale=s, size=(n, l, 2, 2, n))).astype(np.int64)
        H = (H + e) % p

        for i in range(n):
            H[i, :, 0, 1, i] = -1
            H[i, :, 1, 0, i] = -1

        return (H0, H, p, p_exp)

    def eval_seed(self, pp, s, i):
        H0, H, p, p_exp = pp
        l = H0.shape[0]
        n = H.shape[0]
        Z = []
        if i != 0:
            for j in range(l):
                h_sum = 0
                for k in range(n):
                    assert H[i, j, s[i], s[k], k] != -1, f"accessing bot/perp in k={k}, beta={s[k]}"
                    h_sum = (h_sum + int(H[i, j, s[i], s[k], k])) % p
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