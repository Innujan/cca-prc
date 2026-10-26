import galois
import numpy as np
from itertools import combinations

sparse_poly_16 = galois.Poly.Degrees([16, 5, 3, 1, 0], field=galois.GF2)
SGF16 = galois.GF(2**16, irreducible_poly=sparse_poly_16)
sparse_poly_32 = galois.Poly.Degrees([32, 7, 3, 2, 0], field=galois.GF2)
SGF32 = galois.GF(2**32, irreducible_poly=sparse_poly_32)
sparse_poly_64 = galois.Poly.Degrees([64, 4, 3, 1, 0], field=galois.GF2)
SGF64 = galois.GF(2**64, irreducible_poly=sparse_poly_64)
sparse_poly_128 = galois.Poly.Degrees([128, 7, 2, 1, 0], field=galois.GF2)
SGF128 = galois.GF(2**128, irreducible_poly=sparse_poly_128)

poly_16_dense = galois.primitive_poly(2, 16, method="max")
DGF16 = galois.GF(2**16, irreducible_poly=poly_16_dense)
poly_32_dense = galois.primitive_poly(2, 32, method="max")
DGF32 = galois.GF(2**32, irreducible_poly=poly_32_dense)
poly_64_dense = galois.primitive_poly(2, 64, method="max")
DGF64 = galois.GF(2**64, irreducible_poly=poly_64_dense)
poly_128_dense = galois.primitive_poly(2, 128, method="max")
DGF128 = galois.GF(2**128, irreducible_poly=poly_128_dense)

def wt(x):
    if hasattr(x, "vector"):
        return int(np.count_nonzero(x.vector()))
    return bin(int(x)).count("1")

def to_gf(val, field, l):
    val = int(val)
    if l <= 32:
        return field(val)
    else:
        return field(np.array(val, dtype=object))

def run_1flip(l, field, trials, seed):
    rng = np.random.default_rng(seed)
    all_w, maxs, mins, means = [], [], [], []
    for _ in range(trials):
        B = field.Random(low=1, seed=int(rng.integers(1 << 31)))
        B_inv = B ** -1
        weights = []
        for i in range(l):
            e = to_gf(1 << i, field, l)
            prod = e*B_inv
            weight = wt(prod)
            weights.append(weight/l)
            all_w.append(weight/l)
        maxs.append(max(weights))
        mins.append(min(weights))
        means.append(np.mean(weights))

    max_media = np.mean(maxs)
    max_maxs = np.max(maxs)
    min_means = np.mean(mins)
    min_mins = np.min(mins)
    mean = np.mean(means)
    stdev = np.std(all_w)
    return max_media, max_maxs, min_means, min_mins, mean, stdev

def run_nflips(w, l, field, trials, errloops, seed):
    rng = np.random.default_rng(seed)
    all_w, maxs, mins, means = [], [], [], []

    for _ in range(trials):
        B = field.Random(low=1, seed=int(rng.integers(1 << 31)))
        B_inv = B ** -1
        weights = []
        for _ in range(errloops):
            e = 0
            err_idx = rng.choice(l, size=w, replace=False)
            for i in err_idx:
                e |= (1 << int(i))
            e_gf = to_gf(e, field, l)
            prod = e_gf*B_inv
            weight = wt(prod)
            weights.append(weight/l)
            all_w.append(weight/l)
        maxs.append(max(weights))
        mins.append(min(weights))
        means.append(np.mean(weights))

    max_media = np.mean(maxs)
    max_maxs = np.max(maxs)
    min_means = np.mean(mins)
    min_mins = np.min(mins)
    mean = np.mean(means)
    stdev = np.std(all_w)
    return max_media, max_maxs, min_means, min_mins, mean, stdev

def run_worst(l, field, trials, w, seed):
    rng = np.random.default_rng(seed)
    maxs, mins, means = [], [], []

    # Pre-istanziazione dei vettori di base (eseguiti 1 sola volta in tutto il run)
    E = field([to_gf(1 << i, field, l) for i in range(l)])
    
    # Pre-calcola gli indici delle combinazioni
    comb_indices = list(combinations(range(l), w))

    for _ in range(trials):
        B = field.Random(low=1, seed=int(rng.integers(1 << 31)))
        B_inv = B ** -1

        # Calcola solo l moltiplicazioni (128 invece di 8128!)
        P = E * B_inv
        
        # Converte i risultati in int Python puri per combinare con XOR nativo ad altissima velocità
        P_ints = [int(p.item()) if hasattr(p, "item") else int(p) for p in P]

        weights = []
        if w == 1:
            for p_int in P_ints:
                weights.append(p_int.bit_count())
        elif w == 2:
            # Sfrutta (e_i + e_j) * B^-1 = (e_i * B^-1) ^ (e_j * B^-1)
            for i, j in comb_indices:
                weights.append((P_ints[i] ^ P_ints[j]).bit_count())
        else:
            for idx in comb_indices:
                val = 0
                for i in idx:
                    val ^= P_ints[i]
                weights.append(val.bit_count())

        maxs.append(max(weights) / l)
        mins.append(min(weights) / l)
        means.append(np.mean(weights) / l)

    max_media = np.mean(maxs)
    max_maxs = np.max(maxs)
    min_means = np.mean(mins)
    min_mins = np.min(mins)
    mean = np.mean(means)
    stdev = np.std(means)
    return max_media, max_maxs, min_means, min_mins, mean, stdev

def run_onezero(l, field):
    w1 = []
    w0 = []
    B0 = field(1)
    k = l-1
    B1 = to_gf(1 << k, field, l) ** -1
    for i in range(l):
        e = to_gf(1 << i, field, l)
        prod0 = wt(e*B0)
        prod1 = wt(e*B1)
        w0.append(prod0/l)
        w1.append(prod1/l)
    min0 = np.min(w0)
    max0 = np.max(w0)
    mean0 = np.mean(w0)
    stdev0 = np.std(w0)
    min1 = np.min(w1)
    max1 = np.max(w1)
    mean1 = np.mean(w1)
    stdev1 = np.std(w1)
    return min0, max0, mean0, stdev0, min1, max1, mean1, stdev1

for l, field in [(16, DGF16), (32, DGF32), (64, DGF64), (128, DGF128)]:
    print(f"## TESTING IN FIELD F_2^{l} ##")
    print("-- results of run_1flip --")
    max_media, max_maxs, min_means, min_mins, mean, stdev = run_1flip(l, field, 1000, 42)
    print(f"max_media = {max_media}\nmax_maxs = {max_maxs}\nmin_media = {min_means}\nmin_mins = {min_mins}\nmedia = {mean}\nstdev = {stdev}")
    for w in [1, 2]:
        print(f"-- results of run_nflips (w = {w}) --")
        max_media, max_maxs, min_means, min_mins, mean, stdev = run_nflips(w, l, field, 1000, 100, 42)
        print(f"max_media = {max_media}\nmax_maxs = {max_maxs}\nmin_media = {min_means}\nmin_mins = {min_mins}\nmedia = {mean}\nstdev = {stdev}")
        print(f"-- results of run_worst (w = {w}) --")
        max_media, max_maxs, min_means, min_mins, mean, stdev = run_worst(l, field, 1000, w, 42)
        print(f"max_media = {max_media}\nmax_maxs = {max_maxs}\nmin_media = {min_means}\nmin_mins = {min_mins}\nmedia = {mean}\nstdev = {stdev}")
    print(f"-- results of run_onezero --")
    min0, max0, mean0, stdev0, min1, max1, mean1, stdev1 = run_onezero(l, field)
    print(f"min0 = {min0}\nmax0 = {max0}\nmedia0 = {mean0}\nstdev0 = {stdev0}\nmin1 = {min1}\nmax1 = {max1}\nmedia1 = {mean1}\nstdev1 = {stdev1}")
