"""
Flexible-operation planning for the electrochemical-NH3 + DEF plant.

Question: with wind-only electricity, is it cheaper to
  (A) run everything continuously and firm the wind with a battery,
  (B) let the electrochemical NH3 section follow the wind (cells can
      ramp and stop) and buffer NH3 in a tank so the DEF/urea section
      runs continuously, or
  (C) as (B) but the DEF plant (sized 1.2x) may run at 60-120 % of the
      average load
      (same annual output)?
In (B) and (C) the battery is still sized freely (the DEF plant needs some
power through windless hours); "battery" fixes the NH3 section continuous.

Hourly LP over a full year (2024 Hornsea-2 wind profile, cf_wind column):
  decisions  wind capacity W [kW], battery energy B [kWh] and power Pb [kW],
             NH3 tank T [kg], NH3-section capacity factor-up Cap [kg/h NH3]
             (electrode area scales with it), hourly NH3 rate n(t),
             battery charge/discharge, curtailment
  balance    W*cf(t) + dis(t) = ch(t) + curt(t) + P_A(t) + P_U
             P_A(t) = e_A * n(t) - credit_HI * n(t)/Cap... (linearised:
             plant-wide heat-integration credit only while NH3 runs)
             tank: T(t+1) = T(t) + n(t) - n_U ; 0 <= T(t) <= T
             battery: SOC(t+1) = SOC(t) + eta*ch - dis/eta ; SOC <= B
  urea/DEF   constant NH3 draw n_U (continuous; the 150 bar reactor and
             carbamate loop are not ramped), constant power P_U
  objective  annualised cost: wind + battery + tank + cell stack

Energy data come from figures/electro_cases.json (written by
idaes_model/electro_cases.py); costs are the constants below (rough,
documented in docs/electro_def_plant.md).

    python flex_planning.py                  # three strategies
    python flex_planning.py --hours 2000     # shorter horizon, quick
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import idaes  # noqa: F401  (puts the IDAES cbc build on PATH)
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
WIND_CSV = ROOT / "optimization" / "uk_hornsea2_wind_availability_2024.csv"
CASES_JSON = ROOT / "figures" / "electro_cases.json"

# annualised costs ($/yr per unit), capex * CRF + fixed O&M
CRF = lambda i, n: i * (1 + i) ** n / ((1 + i) ** n - 1)  # noqa: E731
COST = {
    "wind_per_kw": 1500 * CRF(0.08, 25) + 45,      # offshore-share PPA proxy
    "batt_per_kwh": 250 * CRF(0.08, 15) + 5,       # Li-ion energy
    "batt_per_kw": 100 * CRF(0.08, 15),            # inverter / power
    "tank_per_kg": 3.0 * CRF(0.08, 25),            # pressurised NH3 bullet
    "cell_per_kgph": None,                         # from stack area, below
}
STACK_PER_M2 = 6000.0      # $/m2 electrode (central), 10 yr life
ETA_BATT = 0.95            # one-way
N_MIN_FRAC = 0.0           # NH3-section turndown (cells can stop)
DEF_MIN_FRAC = 0.5         # DEF plant turndown when allowed (150 bar loop)
DEF_OVERSIZE = 1.2         # DEF design / average load when it may flex
                           # (DEF capex is small next to cells + wind; not costed)


def load(hours):
    cf = pd.read_csv(WIND_CSV)["cf_wind"].clip(0, 1).to_numpy()
    if hours:
        cf = cf[:hours]
    d = json.loads(CASES_JSON.read_text())["electro_plus_central"]
    return cf, d


def build(cf, d, strategy):
    T = range(len(cf))
    m = pyo.ConcreteModel()
    n_u = d["NH3 [kg/h]"]                      # continuous NH3 draw by DEF
    e_a = d["ammonia kW"] / n_u                # kW per kg/h NH3
    p_u = d["urea kW standalone"]              # DEF alone (no HI with NH3)
    credit = d["HI credit kW"] / n_u           # kW saved per kg/h NH3 run
    area_per_kgph = d["cell area m2"] / n_u
    cost_cell = STACK_PER_M2 * CRF(0.08, 10) * area_per_kgph

    m.W = pyo.Var(bounds=(0, None))
    m.B = pyo.Var(bounds=(0, None))
    m.Pb = pyo.Var(bounds=(0, None))
    m.Tk = pyo.Var(bounds=(0, None))
    m.Cap = pyo.Var(bounds=(n_u, None))
    m.n = pyo.Var(T, bounds=(0, None))
    m.ch = pyo.Var(T, bounds=(0, None))
    m.dis = pyo.Var(T, bounds=(0, None))
    m.curt = pyo.Var(T, bounds=(0, None))
    m.soc = pyo.Var(T, bounds=(0, None))
    m.inv = pyo.Var(T, bounds=(0, None))
    # NH3 drawn by the DEF plant: constant unless the DEF plant may turn down
    m.u = pyo.Var(T, bounds=(DEF_MIN_FRAC * DEF_OVERSIZE * n_u,
                             DEF_OVERSIZE * n_u))

    if strategy == "battery":            # everything continuous
        for t in T:
            m.n[t].fix(n_u)
        m.Tk.fix(0.0)
        m.Cap.fix(n_u)
    if strategy != "flex_def":           # DEF plant runs at constant load
        for t in T:
            m.u[t].fix(n_u)

    tl = len(cf)
    m.power = pyo.Constraint(T, rule=lambda m, t:
                             m.W * cf[t] + m.dis[t] == m.ch[t] + m.curt[t]
                             + (e_a - credit) * m.n[t] + p_u * m.u[t] / n_u)
    m.cap = pyo.Constraint(T, rule=lambda m, t: m.n[t] <= m.Cap)
    m.socb = pyo.Constraint(T, rule=lambda m, t:
                            m.soc[(t + 1) % tl] == m.soc[t]
                            + ETA_BATT * m.ch[t] - m.dis[t] / ETA_BATT)
    m.socmax = pyo.Constraint(T, rule=lambda m, t: m.soc[t] <= m.B)
    m.chmax = pyo.Constraint(T, rule=lambda m, t: m.ch[t] <= m.Pb)
    m.dismax = pyo.Constraint(T, rule=lambda m, t: m.dis[t] <= m.Pb)
    m.invb = pyo.Constraint(T, rule=lambda m, t:
                            m.inv[(t + 1) % tl] == m.inv[t] + m.n[t] - m.u[t])
    m.annual = pyo.Constraint(expr=sum(m.u[t] for t in T) == n_u * tl)
    m.invmax = pyo.Constraint(T, rule=lambda m, t: m.inv[t] <= m.Tk)
    scale = 8784.0 / tl
    m.obj = pyo.Objective(expr=(
        COST["wind_per_kw"] * m.W + COST["batt_per_kwh"] * m.B
        + COST["batt_per_kw"] * m.Pb + COST["tank_per_kg"] * m.Tk
        + cost_cell * m.Cap) * 1.0)
    m._meta = dict(n_u=n_u, e_a=e_a, p_u=p_u, credit=credit,
                   cost_cell=cost_cell, scale=scale)
    return m


def run(strategy, cf, d):
    m = build(cf, d, strategy)
    s = pyo.SolverFactory("cbc")
    r = s.solve(m)
    assert str(r.solver.termination_condition) == "optimal", r
    md = m._meta
    T = range(len(cf))
    urea = md["n_u"] / 2 / 17.031 * 60.055 * len(cf)        # kg over horizon
    e_wind = sum(pyo.value(m.W) * cf[t] for t in T)
    e_curt = sum(pyo.value(m.curt[t]) for t in T)
    on = sum(1 for t in T if pyo.value(m.n[t]) > 1e-3) / len(cf)
    res = {
        "strategy": strategy,
        "wind [kW]": pyo.value(m.W),
        "battery [kWh]": pyo.value(m.B),
        "battery power [kW]": pyo.value(m.Pb),
        "NH3 tank [kg]": pyo.value(m.Tk),
        "tank [h of DEF feed]": pyo.value(m.Tk) / md["n_u"],
        "NH3 section capacity [kg/h]": pyo.value(m.Cap),
        "oversize factor": pyo.value(m.Cap) / md["n_u"],
        "NH3 section on-time": on,
        "curtailed [%]": 100 * e_curt / max(e_wind, 1e-9),
        "annualised cost [$/yr]": pyo.value(m.obj),
        "cost [$/kg urea]": pyo.value(m.obj) / (urea * md["scale"]),
        "cost wind [$/yr]": COST["wind_per_kw"] * pyo.value(m.W),
        "cost battery [$/yr]": COST["batt_per_kwh"] * pyo.value(m.B)
        + COST["batt_per_kw"] * pyo.value(m.Pb),
        "cost NH3 tank [$/yr]": COST["tank_per_kg"] * pyo.value(m.Tk),
        "cost NO3RR stack [$/yr]": md["cost_cell"] * pyo.value(m.Cap),
    }
    series = pd.DataFrame({
        "cf": cf, "wind_kw": [pyo.value(m.W) * c for c in cf],
        "nh3_kgph": [pyo.value(m.n[t]) for t in T],
        "soc_kwh": [pyo.value(m.soc[t]) for t in T],
        "tank_kg": [pyo.value(m.inv[t]) for t in T],
        "curt_kw": [pyo.value(m.curt[t]) for t in T],
        "def_nh3_kgph": [pyo.value(m.u[t]) for t in T]})
    return res, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=0)
    a = ap.parse_args()
    cf, d = load(a.hours)
    out, series = [], {}
    for s in ("battery", "flex_nh3", "flex_def"):
        r, ser = run(s, cf, d)
        out.append(r)
        series[s] = ser
        print(f"\n== {s}")
        for k, v in r.items():
            if k != "strategy":
                print(f"  {k:30s} {v:14.3f}")
    outdir = ROOT / "optimization" / "solutions"
    outdir.mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv(outdir / "flex_planning_summary.csv", index=False)
    for s, ser in series.items():
        ser.to_csv(outdir / f"flex_planning_{s}.csv", index=False)
    import flex_plots
    flex_plots.plot(out, series)
    print(f"\nwrote {outdir}/flex_planning_*.csv and figures/flex_planning_*.png")


if __name__ == "__main__":
    main()
