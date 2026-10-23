import hashlib
import secrets
import numpy as np
import os

class DS:
    def __init__(self):
        pass

    @staticmethod
    def setup(self, lam):
        n = 2*lam*lam
        sk = np.random.randint(low=0, high=2, size=(lam, 2, lam), dtype=int)
        h = hashlib.sha256()
        
        vk = np.empty_like(vk)

        for i in range(lam):
            for j in range(2):
                vk[i][j] = bin(int(h(sk[i][j].encode()).hexdigest(), 16))
        
        return [vk, sk]

    @classmethod
    def sign(sk, msg):
        sign = [sk[i][m[i]] for i in range(len(m))]
        return (msg, sign)

    @classmethod
    def verify(vk, msg, sign):
        i = 0
        while bin(int(h(sign[i].encode()).hexdigest(), 16)) == vk[i][m[i]]:
            i+=1
        return True if i == len(sign)-1 else False
