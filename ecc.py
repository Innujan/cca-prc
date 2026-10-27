from pyldpc import make_ldpc, decode, get_message
import numpy as np


class ECC:
    def __init__(self, msg_l, dv=4, dc=8):
        self.l = msg_l
        self.dv = dv
        self.dc = dc

    def setup(self):
        rate = 1 - (self.dv / self.dc)
        n = int(self.l / rate)
        n = -(-n // self.dc) * self.dc  # make_ldpc richiede n multiplo di dc

        H, G = make_ldpc(n, self.dv, self.dc, systematic=True, sparse=True)
        k = G.shape[1]
        assert k >= self.l, "il codice ha dimensione minore del messaggio richiesto"

        return (H, G, self.l, k, n)

    def enc(self, pp, msg):
        H, G, req_k, act_k, n = pp
        msg = np.asarray(msg, dtype=int)
        if len(msg) != req_k:
            raise ValueError(f"message length {len(msg)} != {req_k}")

        pad = act_k - req_k
        msg_padd = np.pad(msg, (0, pad), 'constant', constant_values=0) if pad > 0 else msg

        bin_cw = np.dot(G, msg_padd) % 2
        return bin_cw

    def _get_snr(self, p):
        if p <= 0.0:
            return 30.0
        if p >= 0.5:
            return -20.0

        s_sq = 2.0 / np.log((1 - p) / p)
        snr = 10 * np.log10(1.0 / s_sq)
        return snr

    def dec(self, pp, noisy_cw, p):
        H, G, k_req, k_act, n = pp
        signoise_r = self._get_snr(p)
        bp_signal = 1.0 - 2.0 * np.asarray(noisy_cw)
        # pyldpc 0.7.x: decode restituisce gia' i bit (0/1) della parola di codice
        est_bin_cw = np.asarray(decode(H, bp_signal, signoise_r, maxiter=100)).astype(int)
        synd = np.dot(H, est_bin_cw) % 2
        if np.any(synd != 0):
            raise ValueError("LDPC belief propagation decoding failure")
        est_msg = get_message(G, est_bin_cw)
        return est_msg[:k_req]


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")  # pyldpc avvisa con un warning se BP non converge

    k = 32
    ecc = ECC(msg_l=k, dv=4, dc=8)
    pp = ecc.setup()
    H, G, k_req, k_act, n = pp
    assert k_act >= k_req

    rng = np.random.default_rng(42)
    msg = rng.integers(2, size=k)

    # round trip
    cw = ecc.enc(pp, msg)
    assert len(cw) == n
    np.testing.assert_array_equal(msg, ecc.dec(pp, cw, p=0.0001))
    print("round trip passed")

    # sotto soglia
    noisy = cw.copy()
    noisy[5] ^= 1
    noisy[12] ^= 1
    np.testing.assert_array_equal(msg, ecc.dec(pp, noisy, p=0.05))
    print("low errors passed")

    # sopra soglia: deve fallire (eccezione) oppure, se BP converge a un'altra
    # parola, restituire un messaggio diverso -> mai un successo "spurio" silenzioso
    d_cw = cw.copy()
    err_idx = rng.choice(len(cw), size=int(len(cw) * 0.40), replace=False)
    d_cw[err_idx] ^= 1
    try:
        out = ecc.dec(pp, d_cw, p=0.1)
        assert not np.array_equal(out, msg), "decodifica corretta con il 40% di errori?"
        print("high errors: BP converge a parola sbagliata (da scartare in CheckNoise)")
    except ValueError:
        print("high errors: fallimento rilevato")

    # lunghezza errata del messaggio
    try:
        ecc.enc(pp, msg[:-1])
        raise AssertionError("avrebbe dovuto sollevare ValueError")
    except ValueError:
        print("length check passed")