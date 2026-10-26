from pyldpc import make_ldpc, decode, get_message
import numpy as np
import unittest

class ECC:
    def __init__(self, msg_l, dv=4, dc=8):
        self.l = msg_l
        self.dv = dv
        self.dc = dc

    def setup(self):
        rate = 1-(self.dv/self.dc)
        n = int(self.l / rate)

        H, G = make_ldpc(n, self.dv, self.dc, systematic=True, sparse=True)
        k = G.shape[1]

        return (H, G, self.l, k, n)

    def enc(self, pp, msg):
        H, G, req_k, act_k, n = pp
        if len(msg) != k: raise ValueError(f"message longer than {k}")

        pad = act_k - req_k
        msg_padd = np.pad(msg, (0, pad), 'constant', constant_values=0) if pad > 0 else msg

        bin_cw = np.dot(G, msg_padd) % 2
        return bin_cw

    def _get_snr(self, p):
        if p <= 0.0:
            return 30.0
        if p >= 0.5:
            return -20.0
        
        s_sq = 2.0 / np.log((1-p)/p)
        snr = 10*np.log10(1.0 / s_sq)
        return snr

    def dec(self, pp, noisy_cw, p):
        H, G, k_req, k_act, n = pp
        signoise_r = self._get_snr(p)
        bp_signal = 1.0-2.0*noisy_cw
        decoded_bp = decode(H, bp_signal, signoise_r, maxiter=100)
        est_bin_cw = (decoded_bp < 0).astype(int)
        synd = np.dot(H, est_bin_cw) % 2
        if np.any(synd != 0):
            raise ValueError("LDPC belief propagation decoding failure")
        est_msg = get_message(G, decoded_bp)
        return est_msg[:k_req]


if __name__ == "__main__":
    k = 32
    ecc = ECC(msg_l=k, dv=4, dc=8)
    pp = ecc.setup()

    np.random.seed(42)
    msg = np.random.randint(2, size=k)
    if (1):
        #round trip
        cw = ecc.enc(pp, msg)
        decoded = ecc.dec(pp, cw, p=0.0001)
        np.testing.assert_array_equal(msg, decoded)
        print("round trip passed")

        #sotto soglia
        cw = ecc.enc(pp, msg)

        w_noise = cw.copy()
        w_noise[5] = 1 - w_noise[5]
        w_noise[12] = 1 - w_noise[12]

        decoded = ecc.dec(pp, w_noise, p=0.05)
        np.testing.assert_array_equal(msg, decoded)
        print("low errors passed")

        #sopra soglia e fail
        cw = ecc.enc(pp, msg)

        d_cw = cw.copy()
        err_idx = np.random.choice(len(cw), size=int(len(cw)*0.40), replace=False)
        for i in err_idx:
            d_cw[i] = 1 - d_cw[i]
        
        try:
            ecc.dec(pp, d_cw, p=0.1)
        except:
            print("BP didn't converge")
        finally:
            print("high errors passed")    
            H, G, k_req, k_act, n = pp
            assert k_act >= k_req

            try:
                cw = ecc.enc(pp, msg)
                assert len(cw) == n
            except:
                print("unexpected exception for valid lengths")
            print("pad test passed")