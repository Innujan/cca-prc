from lightecc import LightECC
import galois
import numpy as np
from commitments import Com
from ec_elgamal import EC_ElGamal
from ecc import ECC
from hprg import HPRG
from noisy_mceliece import NoisyMcEliece
from pseudorandom_primitives import PRG, PRF
from sign import LamportSignature

class CCAPRC:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.field = galois.GF(2**n)

    def Setup(lam):
        self.ecc = ECC(16)
        ecc_pp = self.ecc.setup()
        self.hprg = HPRG(lam=lam)
        hprg_pp = self.hprg.setup()
        ecc = LightECC(curve_name="secp256k1")
        curve = ecc.curve
        self.pke = EC_ElGamal(curve=curve)
        self.prc = NoisyMcEliece()
        self.com = Com()
        com_pp = self.com.setup(self.n)
        B = bin(int(self.field.Random()))
        pke_keys = [(self.pke.KeyGen(curve), self.pke.KeyGen(curve)) for _ in range(self.n)]
        prc_keys = [(self.prc.KeyGen(), self.prc.KeyGen()) for _ in range(self.n)]

        pk = (ecc_pp, hprg_pp, B, com_pp, [(keys[0][0], keys[1][0]) for keys in pke_keys], [(keys[0][0], keys[1][0]) for keys in prc_keys])
        sk = ([keys[0][1] for keys in pke_keys], [keys[0][1] for keys in prc_keys])

        return pk, sk
    
    def Encode(self, pk, m):
        ecc_pp, hprg_pp, B, com_pp, pke_keys, prc_keys = pk
        s = np.random.randint(low=0, high=2, size=self.lam)
        t = self.ecc.Encode(ecc_pp, s)
        ds_sk, ds_vk = LamportSignature.setup(bit_length=256)
        K = PRF.setup(lam)
        unifrand = [np.random.randint(low=0, high=2, size=self.lam) for i in range(n)]
        hintrand = self.hprg.eval(hprg_pp, t, 0)
        comms, ops = self.com.commit(com_pp, t, ds_vk)
        pke_ct = []
        for i in range(len(t)):
            ti = self.pke.Encode(pke_keys[i][t[i]], 1 | op[i] | ds_vk, hintrand[i])
            t1_ = self.pke.Encode(pke_keys[i][1-t[i]], 0, unifrand[i])
        comps = [(self.prc.Encode(prc_keys[i][0]))]
