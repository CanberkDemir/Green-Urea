"""
IPPS planning for the electrochemical-NH3 + DEF plant, on the SAME basis as
the existing IPPS model (ipps_green_urea_fixed_operating_point_relu_omlt.py).

Imported from the existing model, not copied:
  * wind data and horizon: read_wind_data(WIND_CSV, N_PERIODS), i.e. the
    2024 Hornsea-2 profile, "random_3day_to_1day" sampling (one random day
    per 10-day block, seed 20260415, 888 h); --full-year uses all 8784 h;
  * costs: CRF, c_wind, c_B, c_NH3, c_grid_default, c_CO2, c_O2_sale, the NH3
    and urea plant-capex correlations; battery and NH3-tank efficiencies;
  * demand D_ann (125 kg urea/day), urea section always on, the four grid
    modes (free_grid, grid_10pct, grid_5pct, wind_only).

What changes (electrochemical route, from figures/electro_cases.json):
  * no water electrolyser, H2 storage, H2 compression or separate stripper
    duty; the NH3 section is the NO3RR cascade + rigorous vacuum stripper +
    rectifier, its electricity linear in the NH3 rate (kW per kg/h NH3,
    heat integration already inside, IDAES/pinch result);
  * the DEF section draws a constant power per kg/h urea;
  * NO3RR stack capex = electrode area x C_STACK_GBP_PER_M2, sized by the
    NH3-section capacity Cap (the cells may run anywhere in [0, Cap]);
  * anode O2 is sold at c_O2_sale like the electrolyser O2 in IPPS.

It is an LP (no on/off binaries: the cells modulate continuously; min-load
and start-up costs are TODO G3), solved with cbc.

    python ipps_electro.py                # 4 grid modes, IPPS 888 h horizon
    python ipps_electro.py --full-year    # 8784 h
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import pandas as pd  # noqa: E402
import pyomo.environ as pyo  # noqa: E402

with contextlib.redirect_stderr(io.StringIO()):
    import ipps_green_urea_fixed_operating_point_relu_omlt as I  # noqa: E402
import idaes  # noqa: E402,F401  (IDAES cbc build on PATH)

ROOT = Path(__file__).resolve().parents[1]
CASES_JSON = ROOT / "figures" / "electro_cases.json"
OUT = ROOT / "optimization" / "solutions"

C_STACK_GBP_PER_M2 = 4700.0    # ~ $6000/m2 (central case in electro_no3rr.py)
MW_NH3, MW_UREA, MW_O2 = 17.031, 60.055, 31.998
GRID_MODES = ("free_grid", "grid_10pct", "grid_5pct", "wind_only")


def plant_data(case):
    d = json.loads(CASES_JSON.read_text())[case]
    nh3, urea = d["NH3 [kg/h]"], d["urea [kg/h]"]
    info = d["info"]
    cell_kw = d["electricity by category [kW]"]["NO3RR cells"]
    # anode O2 for ALL cell current (NO3RR + HER): I / 4F, I = 8F n / FE
    n_nh3 = nh3 / MW_NH3 / 3.6                               # mol/s (product)
    current = 8 * 96485.0 * n_nh3 / info["cell FE"]          # A
    o2_kg_per_kg_nh3 = current / (4 * 96485.0) * MW_O2 * 3.6 / nh3
    return {
        "e_A": d["ammonia kW"] / nh3,                 # kW per kg/h NH3
        "e_U": d["urea kW standalone"] / urea,        # kW per kg/h urea
        "hi_credit": d["HI credit kW"] / nh3,         # kW per kg/h NH3 run
        "area_per_kgph": info["cell area [m2]"] / nh3,
        "nh3_per_urea": 2 * MW_NH3 / MW_UREA,         # stoichiometric
        "co2_per_urea": 44.01 / MW_UREA,
        "o2_per_nh3": o2_kg_per_kg_nh3,
        "cell_kw_per_kgph": cell_kw / nh3,
        "label": d["label"], "kWh/kg urea": d["kWh/kg urea"],
    }


def build(cf, p, grid_mode):
    T = range(len(cf))
    tl = len(cf)
    m = pyo.ConcreteModel()
    f_u = I.D_ann / 8760.0                          # kg/h urea, always on
    n_u = f_u * p["nh3_per_urea"]                   # kg/h NH3 drawn by DEF
    m.W = pyo.Var(bounds=(0, None))
    m.B = pyo.Var(bounds=(0, None))
    m.Tk = pyo.Var(bounds=(0, None))
    m.Cap = pyo.Var(bounds=(n_u, None))
    m.n = pyo.Var(T, bounds=(0, None))
    m.wind = pyo.Var(T, bounds=(0, None))
    m.grid = pyo.Var(T, bounds=(0, None))
    m.ch = pyo.Var(T, bounds=(0, None))
    m.dis = pyo.Var(T, bounds=(0, None))
    m.soc = pyo.Var(T, bounds=(0, None))
    m.chN = pyo.Var(T, bounds=(0, None))
    m.disN = pyo.Var(T, bounds=(0, None))
    m.inv = pyo.Var(T, bounds=(0, None))
    e_net = p["e_A"] - p["hi_credit"]
    load = {t: e_net * m.n[t] + p["e_U"] * f_u for t in T}
    m.wav = pyo.Constraint(T, rule=lambda m, t: m.wind[t] <= cf[t] * m.W)
    m.bal = pyo.Constraint(T, rule=lambda m, t:
                           load[t] + m.ch[t] <= m.wind[t] + m.grid[t] + m.dis[t])
    m.chw = pyo.Constraint(T, rule=lambda m, t: m.ch[t] <= m.wind[t])
    m.socb = pyo.Constraint(T, rule=lambda m, t: m.soc[t] == m.soc[(t - 1) % tl]
                            + I.eta_B_ch * m.ch[t] - m.dis[t] / I.eta_B_dis)
    m.socc = pyo.Constraint(T, rule=lambda m, t: m.soc[t] <= m.B)
    m.chc = pyo.Constraint(T, rule=lambda m, t: m.ch[t] <= m.B)     # 1C, as IPPS
    m.disc = pyo.Constraint(T, rule=lambda m, t: m.dis[t] <= m.B)
    m.cap = pyo.Constraint(T, rule=lambda m, t: m.n[t] <= m.Cap)
    m.nbal = pyo.Constraint(T, rule=lambda m, t:
                            n_u + m.chN[t] == m.n[t] + m.disN[t])
    m.invb = pyo.Constraint(T, rule=lambda m, t: m.inv[t] == m.inv[(t - 1) % tl]
                            + I.eta_N_ch * m.chN[t] - m.disN[t] / I.eta_N_dis)
    m.invc = pyo.Constraint(T, rule=lambda m, t: m.inv[t] <= m.Tk)
    if grid_mode == "wind_only":
        for t in T:
            m.grid[t].fix(0.0)
    elif grid_mode in ("grid_5pct", "grid_10pct"):
        share = 0.05 if grid_mode == "grid_5pct" else 0.10
        m.gshare = pyo.Constraint(expr=sum(m.grid[t] for t in T)
                                  <= share * sum(load[t] + m.ch[t] for t in T))
    ann = 8760.0 / tl
    k_nh3_tpy = I.TON_PER_YEAR_PER_KGPH * m.Cap
    k_urea_tpy = I.TON_PER_YEAR_PER_KGPH * f_u
    m.capex_stack = pyo.Expression(expr=C_STACK_GBP_PER_M2
                                   * p["area_per_kgph"] * m.Cap)
    m.capex_corr = pyo.Expression(expr=I.MILLION * (
        I.c_Nplant_base + I.c_Nplant_scale * k_nh3_tpy
        + I.c_Uplant_base + I.c_Uplant_scale * k_urea_tpy))
    m.capex = pyo.Expression(expr=I.c_wind * m.W + I.c_B * m.B
                             + I.c_NH3 * m.Tk + m.capex_stack)
    m.opex = pyo.Expression(expr=ann * sum(
        I.c_grid_default * m.grid[t] + I.c_CO2 * p["co2_per_urea"] * f_u
        - (I.c_O2_sale * p["o2_per_nh3"] * m.n[t]
           if I.ENABLE_OXYGEN_REVENUE else 0.0) for t in T))
    # the plant-capex correlation is a constant offset for this LP except
    # its small Cap term; it is reported separately (see docs)
    m.obj = pyo.Objective(expr=I.CRF * (m.capex + m.capex_corr) + m.opex)
    m._meta = dict(f_u=f_u, n_u=n_u, ann=ann, tl=tl, load=load)
    return m


def run(cf, p, mode):
    m = build(cf, p, mode)
    r = pyo.SolverFactory("cbc").solve(m)
    assert str(r.solver.termination_condition) == "optimal", r
    v = pyo.value
    md = m._meta
    T = range(md["tl"])
    load = sum(v(md["load"][t]) for t in T)
    wind_av = sum(cf[t] * v(m.W) for t in T)
    urea_y = md["f_u"] * 8760
    ann_ex = I.CRF * v(m.capex) + v(m.opex)          # without correlation
    return {
        "grid_mode": mode,
        "W_cap_kw": v(m.W), "B_cap_kwh": v(m.B), "NH3_cap_kg": v(m.Tk),
        "NO3RR capacity [kg/h NH3]": v(m.Cap),
        "oversize factor": v(m.Cap) / md["n_u"],
        "stack area [m2]": p["area_per_kgph"] * v(m.Cap),
        "NH3 section on-time [%]": 100 * sum(v(m.n[t]) > 1e-4 for t in T)
        / md["tl"],
        "grid share [%]": 100 * sum(v(m.grid[t]) for t in T) / max(load, 1e-9),
        "curtailment [%]": 100 * (1 - sum(v(m.wind[t]) for t in T)
                                  / max(wind_av, 1e-9)),
        "elec intensity [kWh/kg urea]": load * md["ann"] / urea_y,
        "mean load [kW]": load / md["tl"],
        "capex wind [GBP]": I.c_wind * v(m.W),
        "capex battery [GBP]": I.c_B * v(m.B),
        "capex NH3 tank [GBP]": I.c_NH3 * v(m.Tk),
        "capex NO3RR stack [GBP]": v(m.capex_stack),
        "capex plant correlation [GBP]": v(m.capex_corr),
        "opex net [GBP/y]": v(m.opex),
        "annualised cost [GBP/y]": v(m.obj),
        "annualised cost excl. plant correlation [GBP/y]": ann_ex,
        "GBP per kg urea": v(m.obj) / urea_y,
        "GBP per kg urea excl. plant correlation": ann_ex / urea_y,
    }, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-year", action="store_true")
    ap.add_argument("--case", default="electro_plus_central")
    a = ap.parse_args()
    if a.full_year:
        cf = pd.read_csv(I.WIND_CSV)["cf_wind"].clip(0, 1).to_numpy()
    else:
        cf = I.read_wind_data(I.WIND_CSV, I.N_PERIODS)["cf_wind"] \
            .clip(0, 1).to_numpy()
    p = plant_data(a.case)
    print(f"plant data ({p['label']}): {p['kWh/kg urea']:.1f} kWh/kg urea "
          f"at design; e_A {p['e_A']:.1f} kW per kg/h NH3, O2 "
          f"{p['o2_per_nh3']:.2f} kg/kg NH3; horizon {len(cf)} h")
    paper = {x["key"]: x for x in json.loads(
        (ROOT / "figures" / "paper_figures" / "paper_numbers.json").read_text())}
    rows = []
    for mode in GRID_MODES:
        r, _ = run(cf, p, mode)
        old = paper.get(mode, {})
        r["IPPS (H2 route) W_cap_kw"] = old.get("W_cap_kw")
        r["IPPS (H2 route) B_cap_kwh"] = old.get("B_cap_kwh")
        r["IPPS (H2 route) elec intensity"] = old.get(
            "elec_intensity_kwh_per_kg")
        r["IPPS (H2 route) annualised cost"] = old.get(
            "annualized_cost_gbp_per_y")
        rows.append(r)
        print(f"\n== {mode}")
        for k, v in r.items():
            if k != "grid_mode" and v is not None:
                print(f"  {k:48s} {v:16.2f}")
    OUT.mkdir(exist_ok=True)
    tag = "fullyear" if a.full_year else "ipps_horizon"
    pd.DataFrame(rows).to_csv(OUT / f"ipps_electro_{tag}.csv", index=False)
    print(f"\nwrote {OUT / f'ipps_electro_{tag}.csv'}")


if __name__ == "__main__":
    main()
