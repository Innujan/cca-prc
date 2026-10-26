import hashlib
import secrets
import numpy as np
import os

class LamportSignature:
    @staticmethod
    def _hash(data):
        return hashlib.sha256(data).digest()

    @staticmethod
    def setup(n=256):
        sk = [(secrets.token_bytes(32), secrets.token_bytes(32)) for _ in range(n)]
        vk = [(LamportSignature._hash(pair[0]), LamportSignature._hash(pair[1])) for pair in sk]
        return sk, vk

    @staticmethod
    def sign(sk, msg_bytes):
        msg_hash = LamportSignature._hash(msg_bytes)
        bits = []
        for byte in msg_hash:
            for b in range(7, -1, -1):
                bits.append((byte >> b) & 1)
        sign = [sk[i][bits[i]] for i in range(len(bits))]
        return sign

    @staticmethod
    def verify(vk, msg_bytes, sign):
        if len(sign) != len(vk):
            return False
        
        msg_hash = LamportSignature._hash(msg_bytes)
        bits = []
        for byte in msg_hash:
            for b in range(7, -1, -1):
                bits.append((byte >> b) & 1)

        for i in range(len(bits)):
            expected = vk[i][bits[i]]
            actual = LamportSignature._hash(sign[i])
            if not secrets.compare_digest(actual, expected):
                return False
        return True

if __name__ == "__main__":
    sk, vk = LamportSignature.setup(bit_length=256)
    msg = b"test msg for Lamport sign"
    sign = LamportSignature.sign(sk, msg)

    good = LamportSignature.verify(vk, msg, sign)
    print(f"Verifica msg originale: {good}")
    assert good is True

    alt_msg = b"altered msg for test"
    bad = LamportSignature.verify(vk, alt_msg, sign)
    print(f"Verifica msg manomesso: {bad}")
    assert bad is False

    alt_sign = list(sign)
    alt_sign[0] = bytes([alt_sign[0][0] ^ 0xFF]) + alt_sign[0][1:]
    esito_firma_falsa = LamportSignature.verify(vk, msg, alt_sign)
    print(f"Verifica sign alterata: {esito_firma_falsa}")
    assert esito_firma_falsa is False