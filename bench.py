import argparse
import collections
import contextlib
import importlib.util
import io
import json
import os
import pickle
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location("ccaprc", os.path.join(HERE, "cca-prc.py"))
ccaprc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ccaprc)
ccaprc.VERBOSE = False

ACC = collections.defaultdict(float)

def _wrap(obj, name, label):
    f = getattr(obj, name)

    def g(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            ACC[label] += time.perf_counter() - t0
    setattr(obj, name, g)


def instrument(sch):
    for m in ("KeyGen", "Encode", "Decode"):
        _wrap(sch.prc, m, f"PRC.{m.lower()}")
    for m, lab in (("KeyGen", "PKE.keygen"), ("Encode_bits", "PKE.enc"), ("parsed_Decode", "PKE.dec"),
                   ("parsed_Recover", "PKE.recover")):
        _wrap(sch.pke, m, lab)
    _wrap(sch.hprg, "setup", "HPRG.setup")
    _wrap(sch.hprg, "eval_all", "HPRG.eval")
    for m in ("setup", "commit", "open_com"):
        _wrap(sch.com, m, f"Com.{m}")
    for code, lab in ((sch.seed_code, "ECC.seed"), (sch.payload_code, "ECC.payload"),
                      (sch.signature_code, "ECC.sig")):
        _wrap(code, "encode", lab)
        _wrap(code, "decode", lab)
    for m, lab in (("setup", "DS.keygen"), ("sign", "DS.sign"), ("verify_digest", "DS.verify")):
        _wrap(sch.ds, m, lab)
    _wrap(sch.prg, "gen_bits", "PRG")
    prf_eval = ccaprc.PRF.eval

    def prf(*a, **k):
        t0 = time.perf_counter()
        try:
            return prf_eval(*a, **k)
        finally:
            ACC["PRF"] += time.perf_counter() - t0
    ccaprc.PRF.eval = staticmethod(prf)


def phase(fn):
    """(risultato, tempo totale, tempo per componente). Le stampe delle primitive sono soppresse."""
    ACC.clear()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = fn()
    tot = time.perf_counter() - t0
    parts = dict(ACC)
    parts["other"] = max(tot - sum(parts.values()), 0.0)
    return out, tot, {k: round(v, 4) for k, v in sorted(parts.items(), key=lambda kv: -kv[1]) if v >= 1e-3}


def parse_sets(items):
    cfg = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--set", action="append", default=[], help="parametro di build_scheme, chiave=valore JSON")
    ap.add_argument("--steps", default="all")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--reuse-prc-key", action="store_true")
    ap.add_argument("--prc-sample", type=int, default=10)
    ap.add_argument("--workdir", default="/tmp/ccaprc_bench")
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    args = ap.parse_args()
    steps = (["setup", "enc", "dec", "warm", "robust", "reject", "report"] if args.steps == "all"
             else args.steps.split(","))
    overrides = parse_sets(args.set)
    os.makedirs(args.workdir, exist_ok=True)
    base = os.path.join(args.workdir, args.name)
    state = json.load(open(base + ".json")) if os.path.exists(base + ".json") else {}

    def save_state():
        with open(base + ".json", "w") as f:
            json.dump(state, f)

    # Codice Hermitiano: costruire il codice (~40 s) e il decoder di Sage (~35 s alla prima decodifica)
    # costa piu' di un passo; dopo il passo setup li salviamo su disco e i passi successivi li ricaricano.
    herm = overrides.get("prc") == "HERMITIAN"
    cache = base + ".prc"
    prc_obj = None
    if herm and os.path.exists(cache) and "setup" not in steps:
        with open(cache, "rb") as f:
            prc_obj, dec = pickle.load(f)
        prc_obj.code.decode_to_message = dec.decode_to_message
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        sch, cfg, build_times = ccaprc.build_scheme(verbose=False, prc_obj=prc_obj, **overrides)
    t_build = time.perf_counter() - t0
    if herm and prc_obj is None:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            kp, ks = sch.prc.KeyGen()
            w = np.zeros(sch.prc.mbits, dtype=np.uint8)
            assert np.array_equal(sch.prc.Decode(ks, sch.prc.Encode(kp, w, b"warmup")), w)
        state["t_herm_decoder_init"] = round(time.perf_counter() - t0, 2)
        with open(cache, "wb") as f:
            pickle.dump((sch.prc, sch.prc.code.decoder()), f)
    instrument(sch)
    rng = np.random.default_rng(0)
    msg = rng.integers(0, 2, sch.m).astype(np.uint8)
    pk = sk = ct = None

    def load_keys():
        with open(base + ".keys", "rb") as f:
            return pickle.load(f)

    def load_ct():
        with open(base + ".ct", "rb") as f:
            return pickle.load(f)

    if "setup" in steps:
        info = dict(t_build=round(t_build, 2), build_times={k: round(v, 3) for k, v in build_times.items()},
                    setup_extrapolated=False)
        one = None
        if args.reuse_prc_key:
            ts = []
            for _ in range(args.prc_sample):
                t1 = time.perf_counter(); one = sch.prc.KeyGen(); ts.append(time.perf_counter() - t1)
            info.update(setup_extrapolated=True, prc_keygen_ms=round(1000 * statistics.mean(ts), 1))
            sch.prc.KeyGen = lambda: one
        (pk, sk), t_setup, parts = phase(sch.Setup)
        if one is not None:
            full = 2 * sch.n * info["prc_keygen_ms"] / 1000
            t_setup += full - parts.get("PRC.keygen", 0.0)
            parts["PRC.keygen"] = round(full, 3)
        pk_bytes = len(pickle.dumps(pk, protocol=5))
        sk_bytes = len(pickle.dumps(sk, protocol=5))
        if one is not None:                       # pickle memorizza una volta sola l'oggetto ripetuto
            pk_bytes += (2 * sch.n - 1) * len(pickle.dumps(one[0], protocol=5))
            sk_bytes += (sch.n - 1) * len(pickle.dumps(one[1], protocol=5))
        hprg_pp = pk[3]
        flat_len = sch.payload_code.cw_bits + 2 * sch.n * sch.cwbits + sch.signature_code.cw_bits
        info.update(t_setup=round(t_setup, 2), parts_setup=parts, pk_bytes=pk_bytes, sk_bytes=sk_bytes,
                    hprg_bytes=int(hprg_pp[0].nbytes + hprg_pp[1].nbytes))
        state.update(info)
        state.update(n=sch.n, seed_bits=sch.seed_code.mbits, L_pc=sch.prc.mbits, cw_bits=sch.cwbits,
                     prc_blocks=sch.prc.nblocks, prc_t=int(sch.prc.t), a=int(sch.alpha), T=int(sch.err_range),
                     l_c=sch.payload_code.cw_bits, l_sigma=sch.signature_code.cw_bits, l_ct=flat_len,
                     msg_bits=sch.m, frac_prc=2 * sch.n * sch.cwbits / flat_len, cfg=cfg)
        with open(base + ".keys", "wb") as f:
            pickle.dump((pk, sk), f, protocol=5)
        save_state()
        print(f"[{args.name}] setup {t_setup:.1f}s  |ct|={flat_len} bit  |pk|={pk_bytes/2**20:.1f} MB  "
              f"a={sch.alpha} T={sch.err_range}", flush=True)

    need_keys = any(s in steps for s in ("enc", "dec", "warm", "robust", "reject"))
    if need_keys and pk is None:
        pk, sk = load_keys()

    if "enc" in steps:
        ct, t, parts = phase(lambda: sch.Encode(pk, msg))
        state.update(t_enc_cold=round(t, 2), parts_enc_cold=parts)
        with open(base + ".ct", "wb") as f:
            pickle.dump(ct, f, protocol=5)
        save_state()
        print(f"[{args.name}] enc (freddo) {t:.1f}s", flush=True)

    if any(s in steps for s in ("dec", "robust", "reject")) and ct is None:
        ct = load_ct()

    if "dec" in steps:
        out, t, parts = phase(lambda: sch.Decode(pk, sk, ct))
        assert out is not None and np.array_equal(out, msg), "round trip fallito"
        state.update(t_dec_cold=round(t, 2), parts_dec_cold=parts)
        save_state()
        print(f"[{args.name}] dec (freddo) {t:.1f}s", flush=True)

    if "warm" in steps:
        te, td = [], []
        for _ in range(args.reps):
            c2, t, pe = phase(lambda: sch.Encode(pk, msg)); te.append(t)
            out, t, pd = phase(lambda: sch.Decode(pk, sk, c2)); td.append(t)
            assert out is not None and np.array_equal(out, msg)
        state.update(t_enc=round(statistics.median(te), 2), t_dec=round(statistics.median(td), 2),
                     t_enc_all=[round(x, 2) for x in te], t_dec_all=[round(x, 2) for x in td],
                     parts_enc=pe, parts_dec=pd, reps=args.reps)
        save_state()
        print(f"[{args.name}] warm enc {state['t_enc']}s dec {state['t_dec']}s  (reps {args.reps})", flush=True)

    if "robust" in steps:
        x = ccaprc.ct_flat(ct); x[rng.choice(len(x), sch.err_range, replace=False)] ^= 1
        out, t, _ = phase(lambda: sch.Decode(pk, sk, ccaprc.ct_unflat(sch, x)))
        state.update(robust_T_ok=bool(out is not None and np.array_equal(out, msg)), t_dec_T=round(t, 2))
        save_state()
        print(f"[{args.name}] T errori -> {'msg' if state['robust_T_ok'] else 'ERRORE'} ({t:.1f}s)", flush=True)

    if "reject" in steps:
        x = ccaprc.ct_flat(ct); x[rng.choice(len(x), sch.err_range + 1, replace=False)] ^= 1
        out, t1, _ = phase(lambda: sch.Decode(pk, sk, ccaprc.ct_unflat(sch, x)))
        ok1 = out is None
        x = rng.integers(0, 2, len(x)).astype(np.uint8)
        out, t2, _ = phase(lambda: sch.Decode(pk, sk, ccaprc.ct_unflat(sch, x)))
        state.update(reject_T1_ok=ok1, t_dec_T1=round(t1, 2), reject_rand_ok=out is None, t_dec_rand=round(t2, 2))
        save_state()
        print(f"[{args.name}] T+1 -> {'⊥' if ok1 else 'ERRORE'} ({t1:.1f}s), casuale -> "
              f"{'⊥' if out is None else 'ERRORE'} ({t2:.1f}s)", flush=True)

    if "report" in steps:
        row = dict(name=args.name, **state)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if not k.startswith(("parts", "cfg", "build"))}))


if __name__ == "__main__":
    main()
