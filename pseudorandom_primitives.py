from cryptography.hazmat.primitives import hashes
import galois
import math
import hashlib
import hmac
import secrets
import numpy as np

class PRG:
    def generate(self, seed, l, field):
        n_bytes = math.ceil(l / 8)

        hasher = hashes.Hash(hashes.SHAKE256(n_bytes))
        hasher.update(seed)
        xpand_bytes = hasher.finalize()

        v_int = int.from_bytes(xpand_bytes, byteorder='big')
        rem = (8 - (l % 8)) % 8
        v_int >>= rem

        return field(v_int)
    
    def gen_bits(self, seed, l):
        n_bytes = math.ceil(l / 8)
        hasher = hashes.Hash(hashes.SHAKE256(n_bytes))
        hasher.update(seed)
        raw_bytes = hasher.finalize()

        byte_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        bits = np.unpackbits(byte_arr, bitorder='big')
        return bits[:l]


class PRF:
    @staticmethod
    def setup(lam_bytes=16):
        return secrets.token_bytes(lam_bytes)
    
    @staticmethod
    def eval(k, x, l_out, mode="shake"):
        if mode == "shake":
            hasher = hashes.Hash(hashes.SHAKE256(l_out))
            hasher.update(k)
            hasher.update(x)
            return hasher.finalize()
        elif mode == "hmac":
            h = hmac.new(k, x, hashlib.sha256)
            d = h.digest()
            if l_out > len(d):
                raise ValueError("requested length is bigger than HMAC-SHA256 length")
            return d[:l_out]
        else:
            raise ValueError("unsupported mode")

    @staticmethod
    def serialize_input(i, b, idx_bytes=4):
        if b not in (0, 1):
            raise ValueError("b must be a binary value")
        return i.to_bytes(idx_bytes, byteorder='big')+bytes([b])

if __name__ == "__main__":
    lb = 131
    nbytes = (lb + 7)//8
    field = galois.GF(2**lb)
    prg = PRG()

    seed = b"test_seed_round_trip_1234567890"
    b = prg.gen_bits(seed, lb)
    assert len(b) == lb
    print("random generation pass")

    padl = (8 - (lb % 8)) % 8
    padb = np.pad(b, (0, padl), constant_values=0)
    packbytes = np.packbits(padb, bitorder='big').tobytes()
    assert len(packbytes) == nbytes
    print("bit-byte conversion with padding pass")

    rawint = int.from_bytes(packbytes, byteorder='big')
    elem = rawint >> padl
    felem = field(elem)

    recovered = int(felem)
    shifted = recovered << padl
    recoveredbytes = shifted.to_bytes(nbytes, byteorder='big')
    assert recoveredbytes == packbytes
    print("inversion field - int - byte passed")

    recoveredarr = np.frombuffer(recoveredbytes, dtype=np.uint8)
    recbits = np.unpackbits(recoveredarr, bitorder='big')[:lb]
    assert np.array_equal(b, recbits)
    print("inversion byte - bit pass (round-trip passed)")

    i, b = 42, 1
    serial = PRF.serialize_input(i, b, idx_bytes=4)
    print(serial)
    assert len(serial) == 5
    assert serial == b'\x00\x00\x00\x2a\x01'
    print("fixed length serialization pass")

    k = PRF.setup(16)
    out64 = PRF.eval(k, serial, l_out=64, mode="shake")
    assert len(out64) == 64
    print("shake256 extension pass")

    out16 = PRF.eval(k, serial, l_out=16, mode="hmac")
    assert len(out16) == 16
    print("hmac retrocompatibility pass")

    print("PRF serialization and fixed length pass")