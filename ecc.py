import numpy as np
import galois

class ECC:
    def __init__(self, mbits, m, parity):
        if parity < 2 or parity >= 2**m-1:
            raise ValueError("invalid parity")
        self.mbits = mbits
        self.m = m
        self.parity = parity
        self.GF = galois.GF(2**m)
        self.n = 2**m-1
        self.rs = galois.ReedSolomon(self.n, self.n-parity, field=self.GF)
        self.t = parity//2
        self.k_symbol = -(-mbits // m)
        kmax = 2**m-1-parity
        full, rem = divmod(self.k_symbol, kmax)
        self.block_sizes = [kmax]*full+([rem] if rem else [])
        self.cw_bits = (self.k_symbol+parity*len(self.block_sizes))*m
        self.cw_symbols = self.cw_bits // m
        self._pw = 1 << np.arange(m-1, -1, -1)

    def _bits_to_symbols(self, bits):
        return (np.asarray(bits, dtype=np.int64).reshape(-1, self.m)*self._pw).sum(axis=1)

    def _symbols_to_bits(self, symbols):
        s = np.asarray(symbols, dtype=np.int64)
        return ((s[:, None] >> np.arange(self.m-1, -1, -1)) & 1).astype(np.uint8).reshape(-1)

    def setup(self):
        return (self.parity, self.cw_bits, self.mbits, self.k_symbol, self.m, self.block_sizes)

    def encode(self, pp, msg):
        parity, cw_bits, mbits, k, m, block_sizes = pp
        msg = np.asarray(msg, dtype=np.uint8)
        if msg.shape != (mbits,):
            raise ValueError(f"message length mismatch {msg.shape} != {mbits}")

        msg_pad = np.zeros(k*m, dtype=np.uint8)
        msg_pad[:mbits] = msg
        symbols = self._bits_to_symbols(msg_pad)
        out, pos = [], 0
        for ks in block_sizes:
            out.append(np.asarray(self.rs.encode(self.GF(symbols[pos:pos+ks])), dtype=np.int64))
            pos += ks
        return self._symbols_to_bits(np.concatenate(out))

    def decode(self, pp, noisy_cw):
        parity, cw_bits, mbits, k, m, block_sizes = pp
        cw = np.asarray(noisy_cw, dtype=np.uint8)
        if cw.shape != (cw_bits,):
            print(f"length mismatch: given cw len = {cw.shape} != {(cw_bits,)}")
            return None
        symbols = self._bits_to_symbols(cw)
        msg, pos = [], 0
        for j, ks in enumerate(block_sizes):
            n = ks + parity
            try:
                dec, nerr = self.rs.decode(self.GF(symbols[pos:pos + n]), errors=True)
            except:
                print("b")
                return None
            #print(f"RS block {j}: ks={ks}, nerr={nerr}")
            if np.any(np.asarray(nerr) < 0):
                print(f"RS block {j}: decoding failure")
                return None
            msg.append(np.asarray(dec, dtype=np.int64))
            pos += n
        return self._symbols_to_bits(np.concatenate(msg))[:mbits]


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    c = ECC(24, 4, 9)
    pp = c.setup()
    assert (c.cw_bits, c.t) == (60, 4)
    msg = rng.integers(0, 2, 24).astype(np.uint8)
    cw = c.encode(pp, msg)
    assert np.array_equal(c.decode(pp, cw), msg)
    assert np.array_equal(cw[:24], msg)
    for _ in range(50):
        e = cw.copy()
        e[rng.choice(60, 4, replace=False)] ^= 1
        assert np.array_equal(c.decode(pp, e), msg)
    e = cw.copy()
    e[[0, 1, 2, 3]] ^= 1
    assert np.array_equal(c.decode(pp, e), msg)
    print("little RS ok")

    c = ECC(10001, 12, 200)
    pp = c.setup()
    assert len(c.block_sizes) == 1
    c2 = ECC(60000, 12, 776)
    pp2 = c2.setup()
    assert len(c2.block_sizes) == 2
    msg = rng.integers(0, 2, 60000).astype(np.uint8)
    cw = c2.encode(pp2, msg)
    assert len(cw) == c2.cw_bits
    t = c2.t
    e = cw.copy()
    e[rng.choice(len(cw), t, replace=False)] ^= 1
    assert np.array_equal(c2.decode(pp2, e), msg)
    e = cw.copy()
    e[:t] ^= 1
    assert np.array_equal(c2.decode(pp2, e), msg)
    e = cw.copy()
    idx = np.arange(0, 12*t, 12)
    e[idx] ^= 1
    assert np.array_equal(c2.decode(pp2, e), msg)
    e = cw.copy()
    e[rng.choice(len(cw), 12*(t+20), replace=False)] ^= 1
    assert c2.decode(pp2, e) is None
    assert c2.decode(pp2, cw[:-1]) is None
    print("multi-block RS ok")