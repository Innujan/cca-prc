import hashlib
import secrets
import math
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

class WinternitzSignature:
    def __init__(self, w=4, hashf=hashlib.sha256):
        if w not in (1, 2, 4, 8):
            raise ValueError("incorrect value for w")
        self.w = w
        self.b = 1 << w
        self.hashf = hashf
        self.dbytes = hashf().digest_size
        self.dbits = self.dbytes*8

        self.l1 = math.ceil(self.dbits / self.w)
        maxchecksum = self.l1 * (self.b - 1)
        self.l2 = math.floor(math.log2(maxchecksum)/self.w)+1

        self.l = self.l1 + self.l2
        self.vklen = self.l*self.dbits
    
    def _hash(self, data):
        return self.hashf(data).digest()

    def _cat(self, x, steps):
        curr = x
        for _ in range(steps):
            curr = self._hash(curr)
        return curr
    
    def _bytes_to_base_w(self, data):
        val = int.from_bytes(data, byteorder='big')
        l = len(data)*8
        digits = []
        for i in range(self.l1):
            shift = l - (i+1)*self.w
            if shift >= 0:
                digit = (val >> shift) & (self.b-1)
            else:
                digit = (val << (-shift)) & (self.b-1)
            digits.append(digit)
        return digits

    def _get_digits(self, msg_digest):
        digits = self._bytes_to_base_w(msg_digest)
        c_sum = sum((self.b - 1) - d for d in digits)
        c_digits = []
        for i in range(self.l2-1, -1, -1):
            c_digits.append((c_sum >> (i * self.w)) & (self.b-1))
        return digits + c_digits

    def setup(self):
        sk = [(secrets.token_bytes(self.dbytes)) for _ in range(self.l)]
        vk = [(self._cat(sk[i], self.b-1)) for i in range(self.l)]
        return sk, vk

    def sign(self, sk, mbytes):
        if len(sk) != self.l:
            raise ValueError
        
        mhash = self._hash(mbytes)
        digits = self._get_digits(mhash)

        signature = [self._cat(sk[i], digits[i]) for i in range(self.l)]
        return signature

    def verify(self, vk, mbytes, signature):
        if len(vk) != self.l or len(signature) != self.l:
            return False

        mhash = self._hash(mbytes)
        digits = self._get_digits(mhash)

        for i in range(self.l):
            expect = vk[i]
            steps = (self.b-1)-digits[i]
            act = self._cat(signature[i], steps)
            if not secrets.compare_digest(act, expect):
                return False
        return True

    @staticmethod
    def vk_to_bytes(vk):
        return b"".join(vk)

    @staticmethod
    def vk_from_bytes(raw, length=32):
        return [raw[i:i+length] for i in range(0, len(raw), length)]

def run_lamport():
    sk, vk = LamportSignature.setup(n=256)
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

if __name__ == "__main__":
    wots = WinternitzSignature()
    sk, vk = wots.setup()
    msg = b"test payload for PRC construction"
    sig = wots.sign(sk, msg)

    assert wots.verify(vk, msg, sig) is True
    print("correctness test passed")

    bad_msg = b"tampered payload"
    assert wots.verify(vk, bad_msg, sig) is False
    print("tampered message detection passed")

    alt_sig = list(sig)
    alt_sig[0] = bytes([alt_sig[0][0] ^ 0xFF])+alt_sig[0][1:]
    assert wots.verify(vk, msg, alt_sig) is False
    print("tampered signature passed")