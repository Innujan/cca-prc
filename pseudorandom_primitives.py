from cryptography.hazmat.primitives import hashes
import galois
import math
import hashlib
import hmac
import secrets

class PRG:
    def generate(self, seed, l, field):
        n_bytes = math.ceil(l / 8)

        hasher = hashes.Hash(hashes.SHAKE256(n_bytes))
        hasher.update(seed)
        xpand_bytes = hasher.finalize()

        v_int = int.from_bytes(xpand_bytes, byteorder='big')
        v_int &= (1 << l) - 1

        return field(v_int)

class PRF:
    @staticmethod
    def setup(lam_bytes):
        return secrets.token_bytes(lam_bytes)
    
    @staticmethod
    def eval(k, x, l_out):
        h = hmac.new(k, x, hashlib.sha256)
        return h.digest()[:l_out]
