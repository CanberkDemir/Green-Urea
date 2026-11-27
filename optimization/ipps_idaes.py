"""
IPPS for the classical (electrolyser H2) route with IDAES-trained ReLU
surrogates embedded by reluMIP (TODO F4): A ammonia, B NH3 recovery and,
as the urea section, D the low-pressure DEF plant. The ureaF replica (C)
is trained and plotted but not used here: on the IDAES model it never
exceeds ~0.5 kg/h urea at <5 wt% over Fnh3 8-12, Fco2 10-50, so it
cannot meet the IPPS demand (5.2 kg/h, >= 32.5 wt%).

Same structure as ipps_green_urea_fixed_operating_point_relu_omlt.py:
one fixed operating point per section (the surrogate inputs are design
variables shared by all hours), hourly on/off for the ammonia train,
urea always on, wind + grid + 1C battery, electrolyser + H2 storage,
NH3 storage between B and C, the same costs, wind data, horizon sampling,
demand and grid modes (imported, not copied).

Differences from the original IPPS:
  * three surrogates (A ammonia, B NH3 recovery, D DEF urea) trained on the
    IDAES flowsheets (surrogates/idaes_train_relumip.py) instead of two on
    the Aspen grids. The section electricity already includes heat
    integration within each section and heat-pump heating, so the old
    linear 7.1 kWh/kg stripper duty and the eta_rec heat-recovery block
    are replaced by surrogate B;
  * networks are embedded with relumip.AnnModel (big-M, LP bound
    tightening) instead of OMLT;
  * the electrolyser capacity is free (>= the H2 rate of the operating
    point), so H2 storage + oversizing can follow the wind.

    python ipps_idaes.py                  # 4 grid modes, 888 h horizon
    python ipps_idaes.py --modes wind_only
Writes optimization/solutions/ipps_idaes_summary.csv and
ipps_idaes_oppoint.json (read by surrogates/idaes_surrogate_plots.py).
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyomo.environ as pyo  # noqa: E402
import tensorflow as tf  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "surrogates"))
from relumip_compat import AnnModel  # noqa: E402  (relumip + bug fix)

with contextlib.redirect_stderr(io.StringIO()), \
        contextlib.redirect_stdout(io.StringIO()):
    import ipps_green_urea_fixed_operating_point_relu_omlt as I  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NET_DIR = ROOT / "surrogates" / "trained" / "idaes_relumip"
OUT = ROOT / "optimization" / "solutions"
GRID_MODES = ("free_grid", "grid_10pct", "grid_5pct", "wind_only")
O2_PER_H2 = 7.94       # kg O2 per kg H2 from water electrolysis
MW_UREA, MW_CO2 = 60.055, 44.01


def load_nets():
    meta = json.loads((NET_DIR / "metadata.json").read_text())
    nets = {}
    for sec in "ABD":
        for out, d in meta[sec]["outputs"].items():
            nets[sec, out] = tf.keras.models.load_model(NET_DIR / d["file"])
    return meta, nets


def embed(m, nets, sec, inputs, out, var):
    """Embed one network with reluMIP in its own sub-block.

    reluMIP's Pyomo interface writes its components (nodes, cR, ...) onto
    the block that owns the input variables, so two networks on the same
    block overwrite each other. Each network gets a block with local
    copies of the inputs/output, tied to the model variables."""
    blk = pyo.Block()
    m.add_component(f"ann_{sec}_{out}", blk)
    xs = []
    for i, v in enumerate(inputs):
        xv = pyo.Var(bounds=(v.lb, v.ub))
        blk.add_component(f"x{i}", xv)
        xs.append(xv)
    blk.y = pyo.Var(bounds=(var.lb, var.ub))
    ann = AnnModel(nets[sec, out], modeling_language="PYOMO",
                   name=f"ann_{sec}_{out}")
    ann.connect_network_input(blk, xs)
    ann.connect_network_output(blk, [blk.y])
    with contextlib.redirect_stdout(io.StringIO()),             contextlib.redirect_stderr(io.StringIO()):
        ann.embed_network_formulation(
            opt_model=blk, bound_tightening_strategy="LP",
            solver=mip_solver(SOLVER)[1])
    blk.link_in = pyo.Constraint(range(len(xs)),
                                 rule=lambda b, i: xs[i] == inputs[i])
    blk.link_out = pyo.Constraint(expr=blk.y == var)


def build(cf, meta, nets, grid_mode, urea_flex=False):
    T = list(range(len(cf)))
    tl = len(T)
    prev = {t: T[t - 1] for t in T}           # cyclic horizon, as IPPS
    m = pyo.ConcreteModel()
    bA, bB, bD = (meta[s]["input_bounds"] for s in "ABD")
    out_rng = {(s, o): tuple(d["range"]) for s in "ABD"
               for o, d in meta[s]["outputs"].items()}

    def opvar(rng, pad=0.25):
        lo, hi = rng
        span = hi - lo
        return pyo.Var(bounds=(lo - pad * span - 1e-3, hi + pad * span + 1e-3))

    # ---------------- operating point (one per section)
    m.Ft = pyo.Var(bounds=tuple(bA["Ft"]))
    m.Fh2 = pyo.Var(bounds=tuple(bA["Fh2"]))
    m.NA = opvar(out_rng["A", "ammonia_kgph"])
    m.WA = opvar(out_rng["A", "water_kgph"])
    m.EA = opvar(out_rng["A", "electric_kwhph"])
    # B is fed by A: its inputs are A's outputs, limited to B's data box
    m.NA.setlb(max(m.NA.lb, bB["ammonia_kgph"][0]))
    m.NA.setub(min(m.NA.ub, bB["ammonia_kgph"][1]))
    m.WA.setlb(max(m.WA.lb, bB["water_kgph"][0]))
    m.WA.setub(min(m.WA.ub, bB["water_kgph"][1]))
    m.NB = opvar(out_rng["B", "anhydrous_nh3_kgph"])
    m.EB = opvar(out_rng["B", "electric_kwhph"])
    # urea section = DEF plant (D): Fco2 -> urea, NH3 make-up, power
    m.Fco2 = pyo.Var(bounds=tuple(bD["Fco2"]))
    m.U = opvar(out_rng["D", "pure_urea_kgph"])
    m.Fnh3 = opvar(out_rng["D", "nh3_makeup_kgph"])
    m.EC = opvar(out_rng["D", "electric_kwhph"])
    embed(m, nets, "A", [m.Ft, m.Fh2], "ammonia_kgph", m.NA)
    embed(m, nets, "A", [m.Ft, m.Fh2], "water_kgph", m.WA)
    embed(m, nets, "A", [m.Ft, m.Fh2], "electric_kwhph", m.EA)
    embed(m, nets, "B", [m.NA, m.WA], "anhydrous_nh3_kgph", m.NB)
    embed(m, nets, "B", [m.NA, m.WA], "electric_kwhph", m.EB)
    embed(m, nets, "D", [m.Fco2], "pure_urea_kgph", m.U)
    embed(m, nets, "D", [m.Fco2], "nh3_makeup_kgph", m.Fnh3)
    embed(m, nets, "D", [m.Fco2], "electric_kwhph", m.EC)
    # DEF leaves at 32.5 wt% urea by design (>= UREA_PRODUCT_WTFRAC_MIN)

    # ---------------- hourly: ammonia train (A+B) on/off at the op point
    m.y = pyo.Var(T, within=pyo.Binary)
    m.h2 = pyo.Var(T, bounds=(0, m.Fh2.ub))
    m.nb = pyo.Var(T, bounds=(0, m.NB.ub))
    m.eab = pyo.Var(T, bounds=(0, m.EA.ub + m.EB.ub))

    def onoff(name, x_t, x_op, ub):
        setattr(m, name + "_1", pyo.Constraint(T, rule=lambda m, t: x_t[t] <= ub * m.y[t]))
        setattr(m, name + "_2", pyo.Constraint(T, rule=lambda m, t: x_t[t] <= x_op))
        setattr(m, name + "_3", pyo.Constraint(
            T, rule=lambda m, t: x_t[t] >= x_op - ub * (1 - m.y[t])))
    onoff("h2c", m.h2, m.Fh2, m.Fh2.ub)
    onoff("nbc", m.nb, m.NB, m.NB.ub)
    onoff("eabc", m.eab, m.EA + m.EB, m.EA.ub + m.EB.ub)

    # ---------------- hourly: urea (DEF) section. IPPS default: always on.
    # --urea-flex: on/off at its operating point like the ammonia train
    # (the DEF model only converges down to Fco2 = 10 kg/h, 13.7 kg/h
    # urea, 2.6x the IPPS demand, so always-on overproduces)
    m.yU = pyo.Var(T, within=pyo.Binary)
    m.u = pyo.Var(T, bounds=(0, m.U.ub))
    m.fn = pyo.Var(T, bounds=(0, m.Fnh3.ub))
    m.ec = pyo.Var(T, bounds=(0, m.EC.ub))
    m.fc = pyo.Var(T, bounds=(0, m.Fco2.ub))
    for t in T:
        if not urea_flex:
            m.yU[t].fix(1)

    def onoffU(name, x_t, x_op, ub):
        setattr(m, name + "_1", pyo.Constraint(T, rule=lambda m, t: x_t[t] <= ub * m.yU[t]))
        setattr(m, name + "_2", pyo.Constraint(T, rule=lambda m, t: x_t[t] <= x_op))
        setattr(m, name + "_3", pyo.Constraint(
            T, rule=lambda m, t: x_t[t] >= x_op - ub * (1 - m.yU[t])))
    onoffU("uc", m.u, m.U, m.U.ub)
    onoffU("fnc", m.fn, m.Fnh3, m.Fnh3.ub)
    onoffU("ecc", m.ec, m.EC, m.EC.ub)
    onoffU("fcc", m.fc, m.Fco2, m.Fco2.ub)
    m.demand = pyo.Constraint(expr=sum(m.u[t] for t in T)
                              >= I.D_ann * tl / 8760.0)

    # ---------------- electrolyser + H2 storage
    m.Ecap = pyo.Var(bounds=(0, None))
    m.Einc = pyo.Var(bounds=(0, None))
    m.Hcap = pyo.Var(bounds=(0, None))
    m.Pel = pyo.Var(T, bounds=(0, None))
    m.chH = pyo.Var(T, bounds=(0, None))
    m.disH = pyo.Var(T, bounds=(0, None))
    m.invH = pyo.Var(T, bounds=(0, None))
    m.elcap = pyo.Constraint(T, rule=lambda m, t: m.Pel[t] <= m.Ecap)
    m.elmin = pyo.Constraint(expr=m.Ecap >= m.Fh2 * I.e_spec_el / I.eta_el)
    # IPPS: the part of the electrolyser sized by the NH3 rate is inside
    # the NH3-plant correlation; only the increment is costed at c_el
    m.einc = pyo.Constraint(expr=m.Einc >= m.Ecap - m.NA / I.eta_el)
    h2prod = {t: I.eta_el * m.Pel[t] / I.e_spec_el for t in T}
    m.h2bal = pyo.Constraint(T, rule=lambda m, t:
                             h2prod[t] + m.disH[t] == m.h2[t] + m.chH[t])
    m.invHb = pyo.Constraint(T, rule=lambda m, t: m.invH[t] == m.invH[prev[t]]
                             + I.eta_H_ch * m.chH[t] - m.disH[t] / I.eta_H_dis)
    m.invHc = pyo.Constraint(T, rule=lambda m, t: m.invH[t] <= m.Hcap)

    # ---------------- NH3 storage between B and C
    m.Ncap = pyo.Var(bounds=(0, None))
    m.chN = pyo.Var(T, bounds=(0, None))
    m.disN = pyo.Var(T, bounds=(0, None))
    m.invN = pyo.Var(T, bounds=(0, None))
    m.nbal = pyo.Constraint(T, rule=lambda m, t:
                            m.nb[t] + m.disN[t] == m.fn[t] + m.chN[t])
    m.invNb = pyo.Constraint(T, rule=lambda m, t: m.invN[t] == m.invN[prev[t]]
                             + I.eta_N_ch * m.chN[t] - m.disN[t] / I.eta_N_dis)
    m.invNc = pyo.Constraint(T, rule=lambda m, t: m.invN[t] <= m.Ncap)

    # ---------------- power
    m.W = pyo.Var(bounds=(0, None))
    m.B = pyo.Var(bounds=(0, None))
    m.wind = pyo.Var(T, bounds=(0, None))
    m.grid = pyo.Var(T, bounds=(0, None))
    m.ch = pyo.Var(T, bounds=(0, None))
    m.dis = pyo.Var(T, bounds=(0, None))
    m.soc = pyo.Var(T, bounds=(0, None))
    load = {t: m.Pel[t] + m.eab[t] + m.ec[t] + I.alpha_H_ch * m.chH[t]
            + I.alpha_H_dis * m.disH[t] for t in T}
    m.wav = pyo.Constraint(T, rule=lambda m, t: m.wind[t] <= cf[t] * m.W)
    m.bal = pyo.Constraint(T, rule=lambda m, t:
                           load[t] + m.ch[t] <= m.wind[t] + m.grid[t] + m.dis[t])
    m.chw = pyo.Constraint(T, rule=lambda m, t: m.ch[t] <= m.wind[t])
    m.socb = pyo.Constraint(T, rule=lambda m, t: m.soc[t] == m.soc[prev[t]]
                            + I.eta_B_ch * m.ch[t] - m.dis[t] / I.eta_B_dis)
    m.socc = pyo.Constraint(T, rule=lambda m, t: m.soc[t] <= m.B)
    m.chc = pyo.Constraint(T, rule=lambda m, t: m.ch[t] <= m.B)
    m.disc = pyo.Constraint(T, rule=lambda m, t: m.dis[t] <= m.B)
    if grid_mode == "wind_only":
        for t in T:
            m.grid[t].fix(0.0)
    elif grid_mode in ("grid_5pct", "grid_10pct"):
        share = 0.05 if grid_mode == "grid_5pct" else 0.10
        m.gshare = pyo.Constraint(expr=sum(m.grid[t] for t in T)
                                  <= share * sum(load[t] + m.ch[t] for t in T))

    # ---------------- cost (IPPS constants)
    ann = 8760.0 / tl
    m.capex_corr = pyo.Expression(expr=I.MILLION * (
        I.c_Nplant_base + I.c_Nplant_scale * I.TON_PER_YEAR_PER_KGPH * m.NA
        + I.c_Uplant_base + I.c_Uplant_scale * I.TON_PER_YEAR_PER_KGPH * m.U))
    m.capex = pyo.Expression(expr=I.c_wind * m.W + I.c_el * m.Einc
                             + I.c_B * m.B + I.c_H * m.Hcap + I.c_NH3 * m.Ncap)
    m.opex = pyo.Expression(expr=ann * sum(
        I.c_grid_default * m.grid[t] + I.c_CO2 * m.fc[t]
        - (I.c_O2_sale * O2_PER_H2 * h2prod[t]
           if I.ENABLE_OXYGEN_REVENUE else 0.0) for t in T))
    m.obj = pyo.Objective(expr=I.CRF * (m.capex + m.capex_corr) + m.opex)
    m._meta = dict(T=T, ann=ann, load=load)
    return m


def mip_solver(name=None):
    """HiGHS if installed, else Gurobi (or the one named)."""
    for cand in ([name] if name else ["highs", "gurobi"]):
        opt = pyo.SolverFactory(cand)
        try:
            if opt.available(exception_flag=False):
                return cand, opt
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("no MIP solver: install highspy or gurobipy")


SOLVER = None   # set from --solver in main()


def run(cf, meta, nets, mode, time_limit, urea_flex=False):
    m = build(cf, meta, nets, mode, urea_flex)
    name, opt = mip_solver(SOLVER)
    if name == "highs":
        opt.options["time_limit"] = time_limit
        opt.options["mip_rel_gap"] = 1e-3
    else:
        opt.options["TimeLimit"] = time_limit
        opt.options["MIPGap"] = 1e-3
    r = opt.solve(m, load_solutions=False)
    if not r.solution:
        raise RuntimeError(f"{mode}: {r.solver.termination_condition}")
    m.solutions.load_from(r)
    v = pyo.value
    T, ann = m._meta["T"], m._meta["ann"]
    urea_y = sum(v(m.u[t]) for t in T) * ann
    load = sum(v(m._meta["load"][t]) for t in T)
    ann_ex = I.CRF * v(m.capex) + v(m.opex)
    op = {k: v(getattr(m, k)) for k in
          ("Ft", "Fh2", "NA", "WA", "EA", "NB", "EB", "Fnh3", "Fco2",
           "U", "EC")}
    res = {
        "grid_mode": mode,
        "termination": str(r.solver.termination_condition),
        "mip_gap_bound": getattr(r.problem, "lower_bound", None),
        "W_cap_kw": v(m.W), "B_cap_kwh": v(m.B), "E_cap_kw": v(m.Ecap),
        "H2_cap_kg": v(m.Hcap), "NH3_cap_kg": v(m.Ncap),
        "urea_flex": urea_flex,
        "urea section on-time [%]": 100 * sum(v(m.yU[t]) > 0.5 for t in T)
        / len(T),
        "ammonia train on-time [%]": 100 * sum(v(m.y[t]) > 0.5 for t in T)
        / len(T),
        "grid share [%]": 100 * sum(v(m.grid[t]) for t in T) / max(load, 1e-9),
        "elec intensity [kWh/kg urea]": load * ann / urea_y,
        "annualised cost [GBP/y]": v(m.obj),
        "annualised cost excl. plant correlation [GBP/y]": ann_ex,
        "GBP per kg urea": v(m.obj) / urea_y,
        "GBP per kg urea excl. plant correlation": ann_ex / urea_y,
        **{f"op {k}": x for k, x in op.items()},
    }
    hourly = pd.DataFrame({
        "t": T, "cf_wind": [cf[t] for t in T],
        "wind_avail_kw": [cf[t] * v(m.W) for t in T],
        "wind_used_kw": [v(m.wind[t]) for t in T],
        "grid_kw": [v(m.grid[t]) for t in T],
        "batt_ch_kw": [v(m.ch[t]) for t in T],
        "batt_dis_kw": [v(m.dis[t]) for t in T],
        "batt_soc_kwh": [v(m.soc[t]) for t in T],
        "electrolyser_kw": [v(m.Pel[t]) for t in T],
        "ammonia_AB_kw": [v(m.eab[t]) for t in T],
        "urea_D_kw": [v(m.ec[t]) for t in T],
        "ammonia_on": [round(v(m.y[t])) for t in T],
        "urea_on": [round(v(m.yU[t])) for t in T],
        "h2_store_kg": [v(m.invH[t]) for t in T],
        "nh3_store_kg": [v(m.invN[t]) for t in T],
        "urea_kgph": [v(m.u[t]) for t in T],
    })
    res["solver"] = name
    return res, op, hourly


def edge_report(meta, op, tol=0.02):
    """Surrogate inputs whose optimum lies within tol (fraction of range)
    of the sampled box edge: where the IDAES data must be extended."""
    keys = {"A": {"Ft": "Ft", "Fh2": "Fh2"},
            "B": {"ammonia_kgph": "NA", "water_kgph": "WA"},
            "D": {"Fco2": "Fco2"}}
    hits = []
    for sec, km in keys.items():
        for k, ok in km.items():
            lo, hi = meta[sec]["input_bounds"][k]
            x = op[ok]
            if x <= lo + tol * (hi - lo):
                hits.append(f"{sec}.{k} low ({x:.4g} vs {lo:.4g})")
            elif x >= hi - tol * (hi - lo):
                hits.append(f"{sec}.{k} high ({x:.4g} vs {hi:.4g})")
    return "; ".join(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=list(GRID_MODES))
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--days", type=int, default=None,
                    help="first N sampled days only (operating-point "
                         "learning, e.g. 10)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--urea-flex", action="store_true",
                    help="urea section on/off hourly (IPPS: always on)")
    ap.add_argument("--solver", default=None,
                    help="highs or gurobi (default: highs if installed)")
    a = ap.parse_args()
    global SOLVER
    SOLVER = a.solver
    print(f"MIP solver: {mip_solver(SOLVER)[0]}", flush=True)
    cf = I.read_wind_data(I.WIND_CSV, I.N_PERIODS)["cf_wind"]         .clip(0, 1).to_numpy()
    if a.days:
        cf = cf[:24 * a.days]
    print(f"horizon {len(cf)} h", flush=True)
    meta, nets = load_nets()
    OUT.mkdir(exist_ok=True)
    tag = ("_ureaflex" if a.urea_flex else "") +         (f"_{a.days}d" if a.days else "")
    op_file = OUT / f"ipps_idaes_oppoint{tag}.json"
    rows, ops = [], {}
    for mode in a.modes:
        r, op, hourly = run(cf, meta, nets, mode, a.time_limit, a.urea_flex)
        hourly.to_csv(OUT / f"ipps_idaes_hourly{tag}_{mode}.csv", index=False)
        r["on domain edge"] = edge_report(meta, op)
        rows.append(r)
        ops[mode] = op
        print(f"\n== {mode} ({r['termination']})")
        for k, x in r.items():
            if isinstance(x, float):
                print(f"  {k:48s} {x:14.3f}")
        print(f"  on domain edge: {r['on domain edge'] or 'none'}", flush=True)
        # operating point on the surrogate surfaces as soon as it is solved
        op_file.write_text(json.dumps(ops, indent=2))
        pd.DataFrame(rows).to_csv(OUT / f"ipps_idaes_summary{tag}.csv",
                                  index=False)
        if not a.no_plot:
            import subprocess
            subprocess.run([sys.executable, str(ROOT / "surrogates" /
                            "idaes_surrogate_plots.py"), str(op_file)],
                           capture_output=True)
            print("  op point plotted: figures/idaes_surrogate_*.png",
                  flush=True)
    print(f"\nwrote {OUT / f'ipps_idaes_summary{tag}.csv'}")


if __name__ == "__main__":
    main()
