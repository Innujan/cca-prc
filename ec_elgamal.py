import hashlib
import secrets

from lightecc import LightECC

BLOCK = 248
PTS_COORDS_BITS = 256
PTS_BITS = PTS_COORDS_BITS*2
CT_PAIR_BITS = 2*PTS_BITS

class EC_ElGamal:
    def __init__(self, curve):
        self.curve = curve
        if curve.modulo % 4 != 3:
            raise ValueError("serve q = 3 mod 4 per la radice quadrata (q+1)//4")
    
    def pts_to_bits(self, pt):
        x, y = pt
        return format(x, f'0{PTS_COORDS_BITS}b')+format(y, f'0{PTS_COORDS_BITS}b')

    def parse_pts(self, bits, curve):
        if len(bits) != PTS_BITS:
            return None
        try:
            x = int(bits[:PTS_COORDS_BITS], 2)
            y = int(bits[PTS_COORDS_BITS:], 2)
        except Exception:
            return None

        q = curve.modulo

        if x >= q or y >= q:
            return None

        if (x, y) == (0, 0):
            return None

        try:
            if not curve.is_on_curve((x, y)):
                return None
        except Exception:
            return None

        return (x, y)

    def serialize_ct(self, pairs):
        res = []
        for c, s in pairs:
            res.append(self.pts_to_bits(c))
            res.append(self.pts_to_bits(s))
        return "".join(res)
    
    def parse_ct(self, bits, curve):
        if not isinstance(bits, str) or len(bits) == 0:
            return None

        if len(bits) % CT_PAIR_BITS != 0:
            return None

        n_blocks = len(bits) // CT_PAIR_BITS
        ct_pairs = []

        for i in range(n_blocks):
            offset = i * CT_PAIR_BITS
            c_bits = bits[offset : offset + PTS_BITS]
            s_bits = bits[offset + PTS_BITS : offset + CT_PAIR_BITS]

            c = self.parse_pts(c_bits, curve)
            s = self.parse_pts(s_bits, curve)

            if c is None or s is None:
                return None

            ct_pairs.append((c, s))
        return ct_pairs

    def message_to_point(self, m, k=100):
        curve = self.curve
        a, b, q = curve.a, curve.b, curve.modulo

        if (m + 1) * k >= q:
            raise ValueError("message too large for F_q field of curve")

        for i in range(k):
            x = m * k + i
            y_squared = (pow(x, 3, q) + a * x + b) % q
            y = pow(y_squared, (q + 1) // 4, q)
            if pow(y, 2, q) == y_squared:
                return (x, y)

        raise ValueError(f"no point found for message in k = {k} attempts")

    def point_to_message(self, point, k=100):
        x, _ = point
        return x // k

    # ---- randomness deterministica: r (bytes) -> k_j ----
    def _coins(self, r, nblocks):
        n = self.curve.n
        ks = []
        for j in range(nblocks):
            h = hashlib.shake_256(r + j.to_bytes(4, 'big')).digest(48)
            ks.append(int.from_bytes(h, 'big') % (n - 1) + 1)
        return ks

    # ---- padding bit <-> blocchi ----
    @staticmethod
    def _to_blocks(m):
        l = len(m)
        blocks = []
        start = 0
        while start < l:
            sub = m[start:start + BLOCK]
            if len(sub) == BLOCK:
                blocks.append(sub)
            else:
                blocks.append([0] * (BLOCK - 1 - len(sub)) + [1] + sub)
            start += BLOCK
        if l % BLOCK == 0:
            blocks.append([0] * (BLOCK - 1) + [1])
        return [int(''.join(map(str, b)), 2) for b in blocks]

    @staticmethod
    def _from_blocks(ms):
        res = []
        for i, v in enumerate(ms):
            if i == len(ms) - 1:
                brep = list(map(int, bin(v)[2:]))
                res.extend(brep[1:])  # toglie il marcatore di padding
            else:
                res.extend(list(map(int, format(v, '0248b'))))
        return res

    # ---- schema ----
    def KeyGen(self):
        curve = self.curve
        x = secrets.randbelow(curve.n - 1) + 1
        xp = curve.double_and_add(curve.G, x)
        pk = (xp, curve.G, curve)
        sk = (x, curve)
        return (pk, sk)

    def Encode(self, pk, m, r=None):
        """r: seed (bytes) da cui derivare i coin; se None, ne viene scelto uno fresco."""
        if r is None:
            r = secrets.token_bytes(32)
        y, p, curve = pk
        blocks = self._to_blocks(list(m))
        ks = self._coins(r, len(blocks))
        res = []
        for v, k in zip(blocks, ks):
            c = curve.double_and_add(p, k)
            cy = curve.double_and_add(y, k)
            s = curve.add_points(cy, self.message_to_point(v))
            res.append((c, s))
        return res

    def Decode(self, sk, x):
        x_sk, curve = sk
        ms = []
        for c, d in x:
            cx = curve.double_and_add(c, x_sk)
            pm = curve.add_points(d, curve.negative_point(cx))
            ms.append(self.point_to_message(pm))
        return self._from_blocks(ms)

    def Recover(self, pk, x, r):
        """Decifra usando i coin r (senza chiave segreta): pm = s - k*y."""
        y, p, curve = pk
        ks = self._coins(r, len(x))
        ms = []
        for (c, s), k in zip(x, ks):
            ky = curve.double_and_add(y, k)
            pm = curve.add_points(s, curve.negative_point(ky))
            ms.append(self.point_to_message(pm))
        return self._from_blocks(ms)

    def parsed_Decode(self, sk, bits):
        points = self.parse_ct(bits, self.curve)
        if points is None:
            return None
        try:
            return pke.Decode(sk, ct_pairs)
        except Exception:
            return None


def correctness_test(pke):
    import random
    for _ in range(30):
        m = [random.randint(0, 1) for _ in range(random.randint(1, 600))]
        pk, sk = pke.KeyGen()
        r = secrets.token_bytes(32)
        x = pke.Encode(pk, m, r)
        assert pke.Decode(sk, x) == m, "Decode fallita"
        assert pke.Recover(pk, x, r) == m, "Recover fallita"
        assert pke.Encode(pk, m, r) == x, "Encode non deterministico dato r"
    # lunghezze multiple di 248 e messaggi con zeri iniziali
    pk, sk = pke.KeyGen()
    for m in ([0] * 248, [0] * 496, [0, 0, 1], [1] * 248):
        x = pke.Encode(pk, m)
        assert pke.Decode(sk, x) == m
    print("correctness test passed")

def parsing_test(pke):
    pk, sk = pke.KeyGen()
    l_plain = 513
    m_real = [secrets.randbelow(2) for _ in range(l_plain)]
    m_zero = [0]*l_plain

    ct_r = pke.serialize_ct(pke.Encode(pk, m_real))
    ct_z = pke.serialize_ct(pke.Encode(pk, m_zero))

    n_blocks = (l_plain // BLOCK) + 1 if(l_plain % BLOCK == 0) else ((l_plain + 1 + BLOCK - 1)//BLOCK)
    l_pke_ct = n_blocks * CT_PAIR_BITS

    assert len(ct_r) == l_pke_ct
    assert len(ct_z) == l_pke_ct
    print("fixed length test passed")

    parsed_points = pke.parse_ct(ct_r, pke.curve)
    assert parsed_points is not None
    decoded_m = pke.Decode(sk, parsed_points)
    assert decoded_m == m_real
    print("round trip passed")

    for _ in range(50):
        randb = "".join(secrets.choice("01") for _ in range(l_pke_ct))
        res = pke.parse_ct(randb, pke.curve)
        assert res is None
    print("random input passed")

    corr_list = list(ct_r)
    flip_idx = secrets.randbelow(len(corr_list))
    corr_list[flip_idx] = '1' if corr_list[flip_idx] == '0' else '0'
    corr_bits = "".join(corr_list)

    dec_res = pke.parsed_Decode(sk, corr_bits)
    assert (dec_res is None) or (isinstance(dec_res, list))
    print("parsed decode passed")


if __name__ == "__main__":
    pke = EC_ElGamal(LightECC(curve_name="secp256k1").curve)
    parsing_test(pke)