import os
import json

_POLY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lowweight_irreducible.json")
_FIELD_CACHE = {}
 

def clmul(a, b):
    if a.bit_length() < b.bit_length():
        a, b = b, a
    r = 0
    while b:
        lsb = b & -b
        r ^= a * lsb
        b ^= lsb
    return r
 
 
def _spread_byte(x):
    return sum(((x >> i) & 1) << (2 * i) for i in range(8))
 
 
_SPREAD = [_spread_byte(x).to_bytes(2, "little") for x in range(256)]
 
 
def sqr(a):
    if a == 0:
        return 0
    raw = a.to_bytes((a.bit_length() + 7) // 8, "little")
    return int.from_bytes(b"".join(_SPREAD[x] for x in raw), "little")
 
 
def pmod(a, f):
    df = f.bit_length() - 1
    da = a.bit_length() - 1
    while da >= df:
        a ^= f << (da - df)
        da = a.bit_length() - 1
    return a
 
 
def pgcd(a, b):
    while b:
        a, b = b, pmod(a, b)
    return a
 
 
def _reduce_sparse(x, l, low_exps, mask):
    while x >> l:
        hi = x >> l
        x &= mask
        for e in low_exps:
            x ^= hi << e
    return x
 
 
def is_irreducible(exps):
    l = exps[0]
    f = sum(1 << e for e in exps)
    low = list(exps[1:])
    mask = (1 << l) - 1
    u = 2
    for _ in range(l // 2):
        u = _reduce_sparse(sqr(u), l, low, mask)
        if pgcd(f, u ^ 2) != 1:
            return False
    return True
 
 
def find_low_weight_irreducible(l):
    if l < 2:
        raise ValueError("grado troppo piccolo")
    for a in range(1, l // 2 + 1):
        if is_irreducible((l, a, 0)):
            return (l, a, 0)
    for a in range(3, l):
        for b in range(2, a):
            for c in range(1, b):
                if is_irreducible((l, a, b, c, 0)):
                    return (l, a, b, c, 0)
    raise ValueError(f"nessun polinomio di peso <= 5 di grado {l}")
 
 
def _load_cache():
    try:
        with open(_POLY_FILE) as f:
            return {int(k): tuple(v) for k, v in json.load(f).items()}
    except (OSError, ValueError):
        return {}
 
 
def irreducible_poly(l):
    cache = _load_cache()
    if l not in cache:
        cache[l] = find_low_weight_irreducible(l)
        try:
            with open(_POLY_FILE, "w") as f:
                json.dump({str(k): list(v) for k, v in sorted(cache.items())}, f)
        except OSError:
            pass
    return cache[l]
 
 
class GF2m:
    def __init__(self, l, exps=None):
        self.l = l
        self.exps = tuple(exps) if exps is not None else irreducible_poly(l)
        if self.exps[0] != l or any(e >= l for e in self.exps[1:]):
            raise ValueError("polinomio non valido per il grado l")
        self.low = self.exps[1:]
        self.mask = (1 << l) - 1
        self.order = 1 << l
 
    def __call__(self, x):                      # conversione, come galois.GF(...)(x)
        x = int(x)
        if x < 0:
            raise ValueError("elemento negativo")
        return _reduce_sparse(x, self.l, self.low, self.mask)
 
    def mul(self, a, b):
        return _reduce_sparse(clmul(a, b), self.l, self.low, self.mask)
 
    def random(self):
        return secrets.randbits(self.l)
 
    def __repr__(self):
        terms = " + ".join(f"x^{e}" if e > 1 else ("x" if e == 1 else "1") for e in self.exps)
        return f"GF(2^{self.l}) mod {terms}"
 
 
def get_field(l):
    if l not in _FIELD_CACHE:
        _FIELD_CACHE[l] = GF2m(l)
    return _FIELD_CACHE[l]
