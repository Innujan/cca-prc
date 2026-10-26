import math
import random
import numpy as np
import galois
from sage.all import *
import sage.coding.codes_catalog as codes

GF2 = galois.GF(2)

class NoisyMcEliece:
    def __init__(self, n=65, m=255, d=37, rho=1/65, epsilon=3/255, noise=0.01, mode="BCH"):
        self.n = n
        self.m = m
        self.d = d
        self.rho = rho
        self.epsilon = epsilon
        self.mode = mode
        self.noise = noise
        self.rng = np.random.default_rng()

    def get_sample(self, radius, n):
        #print(radius*n)
        r = math.floor(radius*n)
        sphere_sizes = [math.comb(n, k) for k in range(r+1)]
        #print(sphere_sizes)
        k = random.choices(range(r+1), weights=sphere_sizes)[0]
        flips = random.sample(range(n), k)
        #print(flips)
        result = np.zeros(n, dtype=int)
        for idx in flips:
            result[idx] = 1 - result[idx]
        return result.astype(int)

    def get_full_column_rank(self, n, k):
        while True:
            T = GF2.Random((k, 2 * n), seed=int(self.rng.integers(1 << 31)))
            if np.linalg.matrix_rank(T) == 2 * n:
                return T

    def find_row_subset(self, T):
        k, cols = T.shape
        while True:
            R = np.sort(np.array(random.sample(range(k), cols))) #retrieve random 2n rows indices
            try: #probability of finding a nxn invertible random matrix = ~0.29
                # probability of T[R] being invertible = wt(v)/k
                Tinv = np.linalg.inv(T[R]) #tries to invert T on the indices of R
                return R, Tinv
            except np.linalg.LinAlgError:
                continue
    
    def f2_to_f16(self, msg, Fq):
        (a, ) = Fq._first_ngens(1)
        msg16 = []
        for i in range(len(msg)//4):
            w = msg[4*i:4*i+4]
            p16 = Fq(0)
            for j in range(len(w)):
                if w[j] == 1:
                    p16 += a**Integer(j)
            msg16.append(p16)
        return msg16

    def f16_to_f2(self, msg):
        msg2 = []
        for p in msg:
            bits = p.polynomial().list()
            if len(bits) < 4:
                bits.extend([0, ]*(4-len(bits)))
            word = [int(i) for i in bits]
            msg2.extend(word)
        return np.asarray(msg2, dtype=int)

    def KeyGen(self, herm_q=4, herm_m=24):
        code, G, t = None, None, 0
        if self.mode == "BCH":
            code = galois.BCH(self.m, d=self.d)
            G = code.G.T
            t = code.t
        elif self.mode == "HERMITIAN":
            Fq = GF(Integer(herm_q**2), 'a'); (a, ) = Fq._first_ngens(1) #define polynomial ring on the finite field of 4 elements
            A = AffineSpace(Fq, Integer(2), names=('x', 'y',)); (x, y,) = A._first_ngens(2) #define polynomial affine space of two variables polynomials and over the finite field of two elements
            C = Curve(y**Integer(herm_q)+y-x**Integer(herm_q+1))
            F = C.function_field()
            pls = F.places()
            Q, = C.places_at_infinity()
            pls.remove(Q)
            G = Integer(herm_m)*Q

            code = codes.EvaluationAGCode(pls, G)

            G = code.generator_matrix()
            k, n = G.nrows(), G.ncols()
            Gf2 = np.zeros((4*k, 4*n), dtype=int)
            for i in range(k):
                for j in range(4):
                    Gf2[4*i + j] = self.f16_to_f2(a**j * G[i])

            G = GF2(Gf2.transpose())
            t = code.decoder().decoding_radius()
        m_act, k_act = G.shape #get actual shapes of matrix
        assert k_act >= 2 * self.n, "serve k >= 2n"
        perm = self.rng.permutation(m_act) #random permutation P
        T = self.get_full_column_rank(self.n, k_act) #full-column-rank matrix
        E = GF2(self.rng.binomial(1, self.noise, size=(m_act, 2 * self.n))) #bernoulli error matrix
        A = GF2(np.asarray(G @ T)[perm]) + E #A = PGT+E
        R, Tinv = self.find_row_subset(T) #retrieve the rows to invert T and the inverted matrix
        pk = A
        sk = dict(perm=perm, code=code, G=G, T=T, R=R, Tinv=Tinv, n=self.n, m=m_act, t=t, rho=self.rho)
        return pk, sk

    def Encode(self, pk, msg):
        s = self.get_sample(self.rho, self.n) # retrieve s within delta=1 range from 0
        x = GF2(np.concatenate((s, msg)).astype(int)) # concatenate (msg, s) = x
        e = GF2(self.get_sample(self.epsilon, pk.shape[0])) # retrieve e within epislon=3 range from 0
        return pk @ x + e

    def Decode(self, sk, c):
        perm, code, G = sk["perm"], sk["code"], sk["G"]
        n, m, t, rho = sk["n"], sk["m"], sk["t"], sk["rho"]

        bits = np.empty(m, dtype=int)
        bits[perm] = np.asarray(c, dtype=int)
        c_np = GF2(bits)

        #c = GF(np.asarray(c)) 
        #c_np = np.empty_like(np.asarray(c)) #empty array with shape and type as c
        #c_np[perm] = np.asarray(c) #for each permuted index, replace it with the actual index of the encoded word (c_np[perm[i]] = c[i])
        #c_np = GF(c_np) #converts for bch use

        #y, nerr = bch.decode(c_np, errors=True) #decodes with bch

        y = []
        if self.mode == "BCH":
            y, nerr = code.decode(c_np, errors=True)
            if nerr == -1:
                print("too much errors for base ECC")
                return None
        elif self.mode == "HERMITIAN":
            Fq = code.base_field()
            word16 = vector(Fq, self.f2_to_f16(bits, Fq))
            try:
                y16 = code.decode_to_message(word16)
            except Exception as e:
                print(e)
                return None
            y = GF2(self.f16_to_f2(y16))
        if np.count_nonzero(c_np - G @ y) > t: #calculates hamming weight
            print("too much errors for symbol correctness")
            return None
        x = sk["Tinv"] @ y[sk["R"]] #calculates x = T^{-1}y
        if np.any(sk["T"] @ x != y): #verifies if y in Im(T): if Tx = y, then also T[R]x = y[R]
            print("y not in image of T")
            return None
        x = np.asarray(x)
        s, msg = x[:n], x[n:] #retrieve s and msg
        threshold = math.floor(rho*n)
        if np.count_nonzero(s) > threshold or np.count_nonzero(msg) > threshold: #checks weight of both
            print("weight check fail")
            return None
        return msg

def test_coherence():
    herm = get_herm_code(24)
    G = GF(get_f2_matrix(herm).transpose())
    for _ in range(20):
        m = np.random.randint(0, 2, size=76)
        sage_enc = f16_to_f2(herm.encode(vector(Fq, f2_to_f16(m))))
        mat_enc = np.asarray(G @ GF(m), dtype=int)
        print(np.array_equal(sage_enc, mat_enc))

def test():
    p = dict(n=65, m=255, d=37, rho=1/65, epsilon=3/255, noise=0.01, mode="BCH")
    nme = NoisyMcEliece(p["n"], p["m"], p["d"], p["rho"], p["epsilon"], p["noise"], p["mode"])
    pk, sk = nme.KeyGen()
    for _ in range(100):
        m = nme.get_sample(p["rho"], p["n"])
        c = nme.Encode(pk, m)
        m_c = nme.Decode(sk, c)
        print(np.array_equal(m, m_c))

test()