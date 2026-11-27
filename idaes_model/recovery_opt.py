"""Design optimisation of section B (S2 vacuum stripper + 15 bar rectifier)
at the IPPS operating point. Objective: electricity per kg NH3 recovered,
with unrecovered NH3 charged at its production energy
(0.1776 kg H2 / kg NH3 x 50 kWh/kg H2 = 8.9 kWh/kg NH3).

Decision variables: stripper pressure (t_b = Tsat(p) + brine BPE 5.5 K),
theoretical stages, recovery, R/Rmin, rectifier feed temperature, overhead
condensing end temperature, overhead MVR on/off.
Constraints:
  * without MVR the stripper reboiler is driven by the 60 C reactor heat:
    t_b <= 60 - dT_min = 50 C
  * the aqua-ammonia condensate must be a liquid at its final temperature
    and the condensing pressure (Henry NH3 + Raoult water <= P_cond)
  * Underwood valid (0 < x_f < x_D)
Costing (heat pump / CW / chiller, pinch at dT_min) is unchanged: elec()
from surrogates/idaes_data.py.

Run:  ../.venv/Scripts/python.exe recovery_opt.py
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

import ammonia_energy as AE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "surrogates"))
from idaes_data import elec, OUT  # noqa: E402

NH3_OP, H2O_OP = 9.85, 5056.0                # IPPS operating point
E_LOST = 3 * 2.016 / 17.031 * 50.0           # kWh per kg NH3 not recovered


def bubble_ok(x_nh3, t_c, p_bar):
    h = 59.0 * AE.math.exp(4200.0 * (1 / (t_c + 273.15) - 1 / 298.15))
    p_b = x_nh3 / (h * 0.018015) + (1 - x_nh3) * AE.p_sat_water(t_c)
    return p_b <= p_bar


def evaluate(d, nh3=NH3_OP, h2o=H2O_OP, ratio=None, t_s13=None):
    st, shaft, info = AE.recovery_streams("S2", nh3, h2o, h2o * ratio, t_s13,
                                          **d)
    t_b = AE.t_sat_water(d["p_strip"]) + AE.BPE_S13
    x_f = info["overhead NH3 [mol%]"] / 100
    if not 0 < x_f < AE.X_D:
        return None
    if not d["overhead_mvr"] and t_b > 60.0 - 10.0:
        return None
    p_cond = (AE.p_sat_water(t_b + AE.DT_MVR) if d["overhead_mvr"]
              else d["p_strip"])
    if not bubble_ok(x_f, d["t_ov_cond"], p_cond):
        return None
    e = elec(st, shaft, 4.0)
    rec = d["recovery"] * nh3
    return e, (e + (nh3 - rec) * E_LOST) / rec


def main():
    a = pd.read_csv(OUT / "A.csv")
    ratio = float((a.s13_kgph / a.water_kgph).median())
    t_s13 = float(a.t_s13.median())
    base = dict(p_strip=0.08, n_stages=10, recovery=0.99, r_factor=1.5,
                t_rect_feed=150.0, t_ov_cond=10.0, overhead_mvr=False)
    e0, j0 = evaluate(base, ratio=ratio, t_s13=t_s13)
    print(f"baseline: {e0:.1f} kW, {e0 / (0.99 * NH3_OP):.2f} kWh/kg, "
          f"J={j0:.2f}")
    grid = dict(p_strip=[0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.2],
                n_stages=[6, 10, 15, 20, 30], recovery=[0.95, 0.97, 0.99,
                                                        0.995],
                r_factor=[1.1, 1.2, 1.3, 1.5], t_rect_feed=[100, 150, 180,
                                                             190],
                t_ov_cond=[10, 15, 20, 25, 30, 35, 40],
                overhead_mvr=[False, True])
    best = []
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        d = dict(zip(keys, vals))
        r = evaluate(d, ratio=ratio, t_s13=t_s13)
        if r:
            best.append((r[1], r[0], d))
    best.sort(key=lambda x: x[0])
    for j, e, d in best[:8]:
        print(f"J={j:.3f} kWh/kg  E={e:.1f} kW  {d}")
    for mvr in (False, True):
        j, e, d = next(b for b in best if b[2]["overhead_mvr"] is mvr)
        print(f"best with overhead_mvr={mvr}: J={j:.3f} E={e:.1f} {d}")


if __name__ == "__main__":
    main()
