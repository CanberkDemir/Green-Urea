"""
Train ReLU surrogates for IPPS sections A, B, C (and D, the DEF plant) on
the IDAES data (surrogates/idaes_data/{A,B,C,D}.csv, from idaes_data.py) and check every
network inside a reluMIP big-M formulation (Pyomo + Gurobi).

One Keras Sequential ReLU net per output (same convention as
surrogate_functions.py). Input/output standardisation is folded into the
first/last layer, so the saved net maps raw units to raw units and reluMIP
can embed it directly.

Check: for a few points per net, the input is fixed in a Pyomo model, the
network is embedded with relumip.AnnModel(...).embed_network_formulation
(big-M, LP bound tightening), and the MIP output is compared with Keras.

    python idaes_train_relumip.py          # system Python (tensorflow + relumip)
Writes surrogates/trained/idaes_relumip/{A,B,C}_<output>.keras and
metadata.json (bounds, metrics, MIP check).
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyomo.environ as pyo  # noqa: E402
import tensorflow as tf  # noqa: E402
from relumip_compat import AnnModel  # noqa: E402  (relumip + bug fix)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "surrogates" / "idaes_data"
OUT = ROOT / "surrogates" / "trained" / "idaes_relumip"
OUT.mkdir(parents=True, exist_ok=True)

SECTIONS = {
    "A": (["Ft", "Fh2"], ["ammonia_kgph", "water_kgph", "electric_kwhph"]),
    "B": (["ammonia_kgph", "water_kgph"],
          ["anhydrous_nh3_kgph", "electric_kwhph"]),
    "C": (["Fnh3", "Fco2"],
          ["pure_urea_kgph", "product_urea_wtfrac", "electric_kwhph"]),
    # low-pressure DEF plant: the urea section the IPPS actually uses
    # (the ureaF replica C cannot reach the IPPS urea spec)
    "D": (["Fco2"], ["pure_urea_kgph", "nh3_makeup_kgph", "electric_kwhph"]),
}
HIDDEN = (8, 8)
EPOCHS = 3000
SEED = 20261126


def train(x, y):
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-8)
    ym, ys = y.mean(), max(y.std(), 1e-8)
    tf.keras.utils.set_random_seed(SEED)
    net = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(x.shape[1],))]
        + [tf.keras.layers.Dense(n, activation="relu") for n in HIDDEN]
        + [tf.keras.layers.Dense(1, activation="linear")])
    net.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss="mse")
    net.fit((x - xm) / xs, (y - ym) / ys, epochs=EPOCHS,
            batch_size=min(32, len(x)), verbose=0,
            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=200, min_lr=1e-5)])
    w = net.get_weights()
    w[0] = (w[0] / xs[:, None]).astype(np.float32)
    w[1] = (w[1] - (xm / xs) @ net.get_weights()[0]).astype(np.float32)
    w[-2] = (w[-2] * ys).astype(np.float32)
    w[-1] = (w[-1] * ys + ym).astype(np.float32)
    net.set_weights(w)
    return net


def mip_check(net, lo, hi, pts):
    """Max |MIP - Keras| over pts with the net embedded by reluMIP."""
    err = 0.0
    for p in pts:
        m = pyo.ConcreteModel()
        # reluMIP's Pyomo interface needs scalar Vars, not indexed ones
        xs = []
        for i, v in enumerate(p):
            xv = pyo.Var(bounds=(lo[i], hi[i]))
            m.add_component(f"x{i}", xv)
            xv.fix(float(v))
            xs.append(xv)
        m.y = pyo.Var(bounds=(-1e7, 1e7))
        ann = AnnModel(net, modeling_language="PYOMO")
        ann.connect_network_input(m, xs)
        ann.connect_network_output(m, [m.y])
        ann.embed_network_formulation(
            opt_model=m, bound_tightening_strategy="LP",
            solver=pyo.SolverFactory("highs"))
        m.obj = pyo.Objective(expr=m.y)
        pyo.SolverFactory("highs").solve(m)
        k = float(net(np.array([p], dtype=np.float32)).numpy()[0, 0])
        err = max(err, abs(pyo.value(m.y) - k))
    return err


def main():
    meta = {}
    for sec, (xin, yout) in SECTIONS.items():
        f = DATA / f"{sec}.csv"
        if not f.exists():
            print(f"skip {sec}: {f} missing")
            continue
        df = pd.read_csv(f)
        x = df[xin].to_numpy(np.float32)
        lo, hi = x.min(0).tolist(), x.max(0).tolist()
        meta[sec] = {"inputs": xin, "input_bounds": dict(zip(xin, zip(lo, hi))),
                     "n_points": len(df), "outputs": {}}
        rng = np.random.default_rng(0)
        pts = x[rng.choice(len(x), size=min(3, len(x)), replace=False)]
        for out in yout:
            y = df[out].to_numpy(np.float32)
            net = train(x, y)
            pred = net(x).numpy().ravel()
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            r2 = float(1 - np.sum((pred - y) ** 2)
                       / max(np.sum((y - y.mean()) ** 2), 1e-12))
            err = mip_check(net, lo, hi, pts)
            path = OUT / f"{sec}_{out}.keras"
            net.save(path)
            meta[sec]["outputs"][out] = {
                "file": path.name, "rmse": rmse, "r2": r2,
                "range": [float(y.min()), float(y.max())],
                "relumip_vs_keras_max_abs": err}
            print(f"{sec} {out:22s} R2={r2:.4f} RMSE={rmse:.4g} "
                  f"reluMIP|dev|={err:.2e}", flush=True)
    meta["hidden_layers"] = list(HIDDEN)
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
