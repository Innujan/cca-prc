from pyldpc import make_ldpc, decode, get_message
import numpy as np

class ECC:
    def __init__(self, msg_l, dv, dc):
        self.l = msg_l
        self.dv = dv
        self.dc = dc

    @staticmethod
    def setup(rate):
        rate = 1-(dv/dc)
        n = int(self.l // rate)

        H, G = make_ldpc(n, self.dv, self.dc, systematic=True, sparse=True)
        k = G.shape[1]

        return (H, G, k, n)

    @staticmethod
    def enc(pp, msg):
        G = pp[1]
        bin_cw = np.dot(G, msg) % 2
        return bin_cw

    @staticmethod
    def dec(pp, noisy_cw, signoise_r):
        H = pp[0]
        G = pp[1]

        bp_signal = 1.0-2.0*noisy_cw
        decoded_bp = decode(H, bp_signal, signoise_r, maxiter=100)
        est_msg = get_message(G, decoded_bp)

        return est_msg