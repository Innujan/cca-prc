from lightecc import LightECC
import galois
import numpy as np
from commitments import Com
from ec_elgamal import Hashed_ElGamal
from ecc import ECC
from hprg import HPRG
import os
from noisy_mceliece import NoisyMcEliece
from pseudorandom_primitives import PRG, PRF, bits_to_bytes, bytes_to_bits
from sign import WinternitzSignature
import time

def ham_weight(a, b):
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))

class CCAPRC:
    def __init__(self, m, Kbytes, seed_code, hprg, com, pke, prc, ds, payload_code, signature_code, prg):
        self.seed_code = seed_code
        self.hprg = hprg
        self.com = com
        self.pke = pke
        self.prc = prc
        self.cwbits = prc.cwbits
        self.ds = ds
        self.payload_code = payload_code
        self.signature_code = signature_code
        self.prg = prg
        self.n = seed_code.cw_bits
        self.Kbytes = Kbytes
        self.pl = 1+8*com.l_bytes + com.tag_bits
        self.pke_ct_bits = self.pke.ct_len(self.pl)
        self.m = m
        self.alpha = prc.alpha
        self.err_range = (seed_code.parity // 2)*(self.alpha+1)
        
    def Setup(self):
        n = self.n
        pke_keys = [(self.pke.KeyGen(), self.pke.KeyGen()) for _ in range(n)]
        prc_keys = [(self.prc.KeyGen(), self.prc.KeyGen()) for _ in range(n)]

        seed_pp, payload_pp, sigma_pp, hprg_pp, com_pp = self.seed_code.setup(), self.payload_code.setup(), self.signature_code.setup(), self.hprg.setup(), self.com.setup(n)
        pk = (seed_pp, payload_pp, sigma_pp, hprg_pp, com_pp, [(keys[0][0], keys[1][0]) for keys in pke_keys], [(keys[0][0], keys[1][0]) for keys in prc_keys])
        sk = ([keys[0][1] for keys in pke_keys], [keys[0][1] for keys in prc_keys])

        return pk, sk
    
    def Encode(self, pk, m):
        seed_pp, payload_pp, sigma_pp, hprg_pp, com_pp, pke_keys, prc_keys = pk
        msg = np.asarray(m, dtype=np.uint8)
        n = self.n
        
        s = bytes_to_bits(os.urandom((self.seed_code.mbits+7)//8), self.seed_code.mbits)
        t = self.seed_code.encode(seed_pp, s)
        #print("Original t:", t)
        ds_sk, ds_vk = self.ds.setup()
        vk = self.ds.vk_digest(ds_vk)
        K = PRF.setup(self.Kbytes)
        
        hintrand = self.hprg.eval_all(hprg_pp, t)
        r0 = hintrand[0]
        half_lr0 = len(r0) // 2
        r00 = bits_to_bytes(r0[:half_lr0])
        r01 = bits_to_bytes(r0[half_lr0:])

        comms, ops = self.com.commit(com_pp, [int(x) for x in t], vk)
        com_bits = []
        for cm in comms:
            st = format(int(cm), f"0{self.com.l_out}b")
            com_bits.append(np.frombuffer(st.encode(), dtype=np.uint8)-48)

        pl = self.pl
        lprc = self.prc.mbits

        items = np.zeros((n, 2, lprc), dtype=np.uint8)
        for i in range(n):
            ti = int(t[i])
            ti_elem = [1] + bytes_to_bits(ops[i]).tolist() + bytes_to_bits(vk).tolist()
            randomness_ti = bits_to_bytes(hintrand[i+1])
            randomness_nti = os.urandom(32)
            pke_ct_ti = self.pke.Encode_bits(pke_keys[i][ti], ti_elem, randomness_ti)
            pke_ct_nti = self.pke.Encode_bits(pke_keys[i][1-ti], [0]*pl, randomness_nti)
            arr_ct_ti = np.frombuffer(pke_ct_ti.encode(), dtype=np.uint8)-48
            arr_ct_nti = np.frombuffer(pke_ct_nti.encode(), dtype=np.uint8)-48
            items[i, ti] = np.concatenate([arr_ct_ti, com_bits[i]])
            items[i, 1-ti] = np.concatenate([arr_ct_nti, com_bits[i]])
        
        comps = np.zeros((n, 2, self.prc.cwbits), dtype=np.uint8)
        for i in range(n):
            comps[i][0] = self.prc.Encode(prc_keys[i][0], items[i][0], PRF.eval(K, PRF.serialize_input(i, 0), 32))
            comps[i][1] = self.prc.Encode(prc_keys[i][1], items[i][1], PRF.eval(K, PRF.serialize_input(i, 1), 32))

        payload = np.concatenate([msg, bytes_to_bits(K), items.reshape(-1)])
        c = self.prg.gen_bits(r00, self.payload_code.cw_bits) ^ self.payload_code.encode(payload_pp, payload)

        signature = self.ds.sign(ds_sk, bits_to_bytes(np.concatenate([c, comps.reshape(-1)])))
        sigmabits = bytes_to_bits(self.ds.sig_to_bytes(signature))
        sigma = self.prg.gen_bits(r01, self.signature_code.cw_bits) ^ self.signature_code.encode(sigma_pp, sigmabits)

        ciphertext = (c, comps, sigma)
        return ciphertext

    def Find(self, com_pp, seed_pp, sk, comps):
        pke_keys, prc_keys = sk
        n = self.n

        candidate = np.ones(n, dtype=np.uint8)
        decoded = [None]*n
        for i in range(n):
            out = self.prc.Decode(prc_keys[i], comps[i][0])
            if out is not None:
                decoded[i] = out
                pkect, com = out[:self.pke_ct_bits], out[self.pke_ct_bits:]
                parsed_m = self.pke.parsed_Decode(pke_keys[i], (np.asarray(pkect, dtype=np.uint8)+48).tobytes().decode())
                if parsed_m is None or len(parsed_m) != self.pl or parsed_m[0] != 1:
                    opens_zero = False
                else:
                    l_op = self.com.l_bytes * 8
                    op, vk = (bits_to_bytes(parsed_m[1:1 + l_op]), bits_to_bytes(parsed_m[1 + l_op:]))
                    opens_zero = None
                    try:
                        com = self.com.field(int((np.asarray(com, dtype=np.uint8)+48).tobytes().decode(), 2))
                        opens_zero = bool(self.com.open_com(com_pp, i, 0, com, op, vk))
                    except Exception:
                        opens_zero = False
                    if opens_zero:
                        candidate[i] = 0
        s = self.seed_code.decode(seed_pp, candidate)
        if s is None:
            return None
        t = self.seed_code.encode(seed_pp, s)
        return t, decoded

    def CheckNoise(self, pk, c, comps, sigma, t, decoded):
        _, payload_pp, sigma_pp, hprg_pp, _, _, prc_pkeys = pk
        
        hintrand = self.hprg.eval_all(hprg_pp, t)
        r0 = hintrand[0]
        half_lr0 = len(r0) // 2
        r00 = bits_to_bytes(r0[:half_lr0])
        r01 = bits_to_bytes(r0[half_lr0:])

        payloadmask = self.prg.gen_bits(r00, self.payload_code.cw_bits)
        payload = self.payload_code.decode(payload_pp, c ^ payloadmask)
        if payload is None: 
            print("payload error")
            return None

        msg = payload[:self.m]
        K = bits_to_bytes(payload[self.m:self.m+8*self.Kbytes])
        items=payload[self.m+8*self.Kbytes:].reshape(self.n, 2, self.prc.mbits)
        
        sigmamask = self.prg.gen_bits(r01, self.signature_code.cw_bits)
        sigmabits = self.signature_code.decode(sigma_pp, sigma^sigmamask)
        if sigmabits is None: 
            print("signature error")
            return None

        cp = payloadmask ^ self.payload_code.encode(payload_pp, payload)
        sigmap = sigmamask ^ self.signature_code.encode(sigma_pp, sigmabits)
        
        compsp = np.empty_like(comps)
        for i in range(self.n):
            compsp[i, 0] = self.prc.Encode(prc_pkeys[i][0], items[i, 0], PRF.eval(K, PRF.serialize_input(i, 0), 32))
            compsp[i, 1] = self.prc.Encode(prc_pkeys[i][1], items[i, 1], PRF.eval(K, PRF.serialize_input(i, 1), 32))
        
        if ham_weight(c, cp) + ham_weight(sigma, sigmap) + ham_weight(comps, compsp) > self.err_range:
            print("too much errors")
            return None
        for i in range(self.n):
            if all([ham_weight(comps[i, b], compsp[i, b])<=self.alpha for b in (0, 1)]):
                if decoded[i] is None or not np.array_equal(decoded[i], items[i, 0]):
                    print("2nd check fail")
                    return None
        
        return msg, K, cp, compsp, items, sigmabits

    def CheckPKE(self, hprg_pp, com_pp, pke_pkeys, t, items):
        hintrand = self.hprg.eval_all(hprg_pp, t)

        vks = set()
        for i in range(self.n):
            b = int(t[i])
            pke_ct_iti = (np.asarray(items[i, b][:self.pke_ct_bits], dtype=np.uint8)+48).tobytes().decode()
            com_i = items[i, b][self.pke_ct_bits:]
            r_i = bits_to_bytes(hintrand[i+1])
            recovered = self.pke.parsed_Recover(pke_pkeys[i][b], pke_ct_iti, r_i)
            if recovered is None or len(recovered) != self.pl or recovered[0] != 1:
                print("pke recover error")
                return None
            l_op = self.com.l_bytes * 8
            parsed = (bits_to_bytes(recovered[1:1+l_op]), bits_to_bytes(recovered[1+l_op:]))
            if parsed is None: 
                print("bytes conversion error")
                return None
            op, vk = parsed
            opens_b = None
            try:
                com = self.com.field(int((np.asarray(com_i, dtype=np.uint8)+48).tobytes().decode(), 2))
                opens_b = bool(self.com.open_com(com_pp, i, b, com, op, vk))
            except Exception as e:
                print("commitment scheme open exception:", e)
                opens_b = False
            if not opens_b: 
                print("failed to assign value to opens_b")
                return None
            if self.pke.Encode_bits(pke_pkeys[i][b], recovered, r_i) != pke_ct_iti:
                print("encoding mismatch")
                return None
            vks.add(vk)
        if len(vks) != 1: 
            print("found multiple different verification keys")
            return None
        vk = vks.pop()
        return vk


    def Decode(self, pk, sk, ciphertext):
        seed_pp, _, _, hprg_pp, com_pp, pke_pkeys, _ = pk
        _, prc_skeys = sk

        c, comps, sigma = ciphertext
        find_res = self.Find(com_pp=com_pp, seed_pp=seed_pp, sk=sk, comps=comps)
        if find_res is None:
            return None
        t, decoded = find_res
        #print("Recovered t:", t)

        checknoise_res = self.CheckNoise(pk, c=c, comps=comps, sigma=sigma, t=t, prc_skeys=prc_skeys, decoded=decoded)
        if checknoise_res is None:
            return None
        msg, K, cp, compsp, items, sigmabits = checknoise_res

        vk = self.CheckPKE(hprg_pp=hprg_pp, com_pp=com_pp, pke_pkeys=pke_pkeys, t=t, items=items)
        if vk is None: return None

        signature = self.ds.sig_from_bytes(bits_to_bytes(sigmabits))
        if signature is None:
            print("signature error")
            return None
        ds_msg = bits_to_bytes(np.concatenate([cp, compsp.reshape(-1)]))
        if not self.ds.verify_digest(vk, ds_msg, signature):
            print("message verification failed")
            return None
        return msg

def run_tests(rt=True, randomness=True, robustness=True, mode="BCH"):
    rng = np.random.default_rng(0)
    m = 128
    Kbytes = 16
    noise = 0.01 if mode=="BCH" else 0.0001
    print("# preparing primitives...")

    print("|# creating seed code...")
    t0 = time.time(); seed_code = ECC(mbits=24, m=4, parity=9); tsc = time.time() - t0
    print(f"|# created seed code in {round(tsc, 4)}s")
    
    print("|# creating hprg...")
    t0 = time.time(); hprg = HPRG(lam=6, l=256, d=4, esp=0.5, k=24, n=60); thprg = time.time() - t0
    print(f"|# created hprg in {round(thprg, 4)}s")
    
    print("|# creating commitments scheme...")
    t0 = time.time(); com = Com(l_bytes=16, tag_bits=256); tcom = time.time() - t0
    print(f"|# created commitment scheme in {round(tcom, 4)}s")
    
    print("|# creating EC-ElGamal PKE scheme...")
    t0 = time.time(); pke = Hashed_ElGamal(LightECC(curve_name="secp256k1").curve); tpke = time.time() - t0
    print(f"|# created PKE scheme in {round(tpke, 4)}s")
    
    print("|# creating Noisy McEliece PRC scheme...")
    t0 = time.time(); prc = NoisyMcEliece(1282, n=65, m=255, d=37, rho=1/65, epsilon=3/255, noise=noise, mode=mode); tprc = time.time() -t0
    print(f"|# created PRC scheme in {round(tprc, 4)}s")

    print("|# creating Winternitz one-time signature scheme...")
    t0 = time.time(); ds = WinternitzSignature(w=8); tds = time.time() - t0
    print(f"|# created signature scheme in {round(tds, 4)}s")
    
    ecc_parity = (9//2)*(prc.alpha+1)
    
    print("|# creating payload code...")
    t0 = time.time(); payload_code = ECC(mbits=154096, m=12, parity=ecc_parity); tpc = time.time() - t0
    print(f"|# created payload code in {round(tpc, 4)}s")
    
    print("|# creating signature code...")
    t0 = time.time(); signature_code = ECC(mbits=ds.sig_bits, m=12, parity=ecc_parity); tsigc = time.time() - t0
    print(f"|# created seed code in {round(tsigc, 4)}s")


    print("|# creating PRG...")
    t0 = time.time(); prg = PRG(); tprg = time.time() - t0
    print(f"|# created PRG in {round(tprg, 4)}s")
    
    print("# creating PRC scheme...")
    t0 = time.time(); scheme = CCAPRC(m, Kbytes, seed_code, hprg, com, pke, prc, ds, payload_code, signature_code, prg); tscheme = time.time() - t0
    print(f"# created PRC scheme in {round(tscheme, 4)}s")

    print("# setup...")
    t0 = time.time(); pk, sk = scheme.Setup(); tset = time.time() - t0
    print(f"# setup completed in {round(tset, 4)}s")
    
    msg = rng.integers(0, 2, 128).astype(np.uint8)
    
    print("# encode...")
    t0 = time.time(); ct = scheme.Encode(pk, msg); tenc = time.time() - t0
    print(f"# encode completed in {round(tenc, 4)}s")
    
    if rt:
        print("# decode...")
        t0 = time.time(); out = scheme.Decode(pk, sk, ct); tdec = time.time() - t0
        print(f"# decode completed in {round(tdec, 4)}s")

        assert out is not None and np.array_equal(msg, out)
        print("round trip ok")
    if randomness:
        print("# encode 2...")
        ct2 = scheme.Encode(pk, msg)
        ct1_flat = np.concatenate([ct[0], ct[1].reshape(-1), ct[2]]).astype(np.uint8)
        ct2_flat = np.concatenate([ct2[0], ct2[1].reshape(-1), ct2[2]]).astype(np.uint8)
        assert not np.array_equal(ct1_flat, ct2_flat)
        print("random codewords ok")
    if robustness:
        flatct = np.concatenate([ct[0], ct[1].reshape(-1), ct[2]]).astype(np.uint8)
        l = len(flatct)
        lc = payload_code.cw_bits
        cw = scheme.cwbits  
        ls = signature_code.cw_bits  
        n = scheme.n

        x = flatct.copy()
        x[np.asarray(rng.choice(l, scheme.err_range, replace=False))] ^= 1
        random_err_ct = (x[:lc].copy(), x[lc:lc+2*n*cw].reshape(n, 2, cw).copy(), x[lc+2*n*cw:].copy())
        out = scheme.Decode(pk, sk, random_err_ct)
        assert np.array_equal(out, msg)
        print("random errors under decoding radius ok")

        x = flatct.copy()
        s0 = lc
        e0 = s0 + cw
        s1 = lc + 1*cw
        e1 = s1 + cw
        x[np.asarray(s0 + rng.choice(e0-s0, scheme.alpha, replace=False))] ^= 1
        x[np.asarray(s1 + rng.choice(e1-s1, scheme.alpha, replace=False))] ^= 1
        random_comps_ct = (x[:lc].copy(), x[lc:lc+2*n*cw].reshape(n, 2, cw).copy(), x[lc+2*n*cw:].copy())
        out = scheme.Decode(pk, sk, random_comps_ct)
        assert np.array_equal(out, msg)
        print("random errors under ECC decoding radius ok")

        x = flatct.copy()
        x[np.asarray(rng.choice(l, scheme.err_range+1, replace=False))] ^= 1
        more_errors = (x[:lc].copy(), x[lc:lc+2*n*cw].reshape(n, 2, cw).copy(), x[lc+2*n*cw:].copy())
        out = scheme.Decode(pk, sk, more_errors)
        assert not np.array_equal(out, msg) and out is None
        print("random errors over decoding radius ok")

        x = flatct.copy()
        x[s0:e1] = np.concatenate([x[s1:e1], x[s0:e0]])
        swapped_comps = (x[:lc].copy(), x[lc:lc+2*n*cw].reshape(n, 2, cw).copy(), x[lc+2*n*cw:].copy())
        out = scheme.Decode(pk, sk, swapped_comps)
        assert not np.array_equal(out, msg) and out is None
        print("swapped PRC components ok")

        x = rng.integers(0, 2, l)
        random_ct = (x[:lc].copy(), x[lc:lc+2*n*cw].reshape(n, 2, cw).copy(), x[lc+2*n*cw:].copy())
        out = scheme.Decode(pk, sk, random_ct)
        assert not np.array_equal(out, msg) and out is None
        print("random codeword reject ok")

run_tests(randomness=False, robustness=False, mode="HERMITIAN")
