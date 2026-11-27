"""
Combined B-lite + D: skip the 15.5 bar anhydrous-NH3 rectifier and feed
aqua ammonia to the DEF plant's LP carbamate condenser.

Current B + D (reference)
  B: vacuum stripper 0.06 bar + overhead MVR (B_DESIGN) -> 15.5 bar
     rectifier (198 C sump) -> anhydrous NH3 -> D's HP NH3 pump.
B-lite + D
  B-lite: same stripper + MVR -> 2 bar rectifier (120 C sump, heat-pump
     reachable; aqua condenser ~50 C on cooling water) -> aqua ammonia at
     mass fraction W -> D's LP carbamate condenser (def_lowp aqua_w option).
  The aqua water enters the recycle (lowers reactor conversion per pass)
  but most of it can leave with the DEF product (67.5 wt% water); the demin
  dilution water drops accordingly. If the aqua brings more water than the
  DEF can carry, the DEF spec becomes infeasible (water make-up >= 0).

D: optimized design (stripping steam 0.2 kg/kg CO2, N/C 3.5), Fco2 chosen
for the IPPS demand (default 3.82 kg/h -> 5.21 kg/h urea).
Electricity: one pinch over B + D (dT_min 10 K), heat pump / cooling water
/ chiller as in surrogates/idaes_data.elec, plus all shaft work.

    python blite_def.py                       # W = 0.35, 0.30, 0.25
    python blite_def.py --w 0.30 --fco2 3.82
Writes figures/blite_def.csv.
"""

import argparse
import contextlib
import io
import logging
import math
import sys
from pathlib import Path

import pandas as pd
from pyomo.environ import value

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "surrogates"))
logging.disable(logging.WARNING)

import ammonia_energy as AE  # noqa: E402
import def_lowp_flowsheet as Dm  # noqa: E402
import plant_energy as P  # noqa: E402
import solver_path  # noqa: E402
from heat_integration import Stream  # noqa: E402
from idaes.core.util.model_serializer import from_json, to_json  # noqa: E402
from idaes_data import B_DESIGN, _converged, elec  # noqa: E402

P_AQ = 2.0            # bar, aqua rectifier
T_AQ_SUMP = 120.2     # C, water Tsat at 2 bar
T_AQ_COND = 50.0      # C, aqua (25-35 wt%) total condenser at 2 bar, CW
T_AQ_FEED = 100.0     # C, rectifier feed after feed/bottoms exchange
R_FACTOR_AQ = 1.2
DH_NH3, DH_H2O = 23.4, 42.0      # kJ/mol near 50-60 C
WATER_PER_S13 = None             # from A.csv medians (as section_b)
OUT = ROOT / "figures" / "blite_def.csv"


def x_mol(w):
    return (w / 17.031) / (w / 17.031 + (1 - w) / 18.015)


def b_streams(nh3_aq_kgph, water_kgph, s13_kgph, t_s13, aqua_w=None):
    """Stripper + MVR (B_DESIGN), then either the 15.5 bar anhydrous
    rectifier (aqua_w None) or a 2 bar aqua rectifier to mass fraction
    aqua_w. Returns (streams, shaft, info)."""
    st, shaft, info = AE.recovery_streams("S2", nh3_aq_kgph, water_kgph,
                                          s13_kgph, t_s13, **B_DESIGN)
    if aqua_w is None:
        return st, shaft, info
    st = [s for s in st if not s.name.startswith("rectifier")]
    rec = B_DESIGN["recovery"]
    n_nh3 = rec * nh3_aq_kgph / 17.031                     # kmol/h
    v = info["stripping vapour [kg/h]"] / 18.015           # kmol/h
    w_ov = max(v - n_nh3, 0.0)
    x_f = n_nh3 / (n_nh3 + w_ov)
    x_d = x_mol(aqua_w)
    if x_d <= x_f:          # overhead already richer: no rectifier needed
        return st, shaft, {**info, "aqua R": 0.0,
                           "aqua water [kg/h]": w_ov * 18.015}
    a = AE.ALPHA_RECT
    r_min = (1 / (a - 1)) * (x_d / x_f - a * (1 - x_d) / (1 - x_f))
    r = R_FACTOR_AQ * max(r_min, 0.1)
    d = n_nh3 / x_d                                        # kmol/h
    lam = x_d * DH_NH3 + (1 - x_d) * DH_H2O                 # kJ/mol
    q_cond = (r + 1) * d * lam * 1e3 / 3600.0              # kW
    feed_kg = n_nh3 * 17.031 + w_ov * 18.015
    bot_kg = (w_ov - d * (1 - x_d)) * 18.015
    q_reb = q_cond + feed_kg * AE.CP_WATER * (T_AQ_SUMP - T_AQ_FEED) / 3600
    st += [
        Stream("aqua rectifier feed preheat", 25.0, T_AQ_FEED,
               feed_kg * AE.CP_WATER * (T_AQ_FEED - 25.0) / 3600,
               section="recovery"),
        Stream("aqua rectifier reboiler (2 bar)", T_AQ_SUMP,
               T_AQ_SUMP + 0.1, q_reb, section="recovery"),
        Stream("aqua rectifier condenser", T_AQ_COND + 0.1, T_AQ_COND,
               q_cond, section="recovery"),
        Stream("aqua rectifier bottoms cooler", T_AQ_SUMP, AE.T_CW + 5.0,
               max(bot_kg, 0) * AE.CP_WATER * (T_AQ_SUMP - AE.T_CW - 5.0)
               / 3600, section="recovery"),
    ]
    return st, shaft, {**info, "aqua R": r,
                       "aqua water [kg/h]": d * (1 - x_d) * 18.015}


# ------------------------------------------------------------- D solves
class DWalker:
    """Converged D model at the optimized design; walks Fco2 / aqua."""

    def __init__(self, aqua_w=None):
        from idaes_data import (D_DESIGN_FCO2, D_NC, D_STEAM_PER_CO2,
                                FCO2_DEF_START)
        self.ctx = solver_path.ipopt314()
        self.ctx.__enter__()
        m = self.m = Dm.build_model(fco2_kgph=FCO2_DEF_START,
                                    direct_frac=0.0, aqua_w=aqua_w)
        with contextlib.redirect_stdout(io.StringIO()):
            self.s = Dm.initialize(m)
            Dm.solve(m, self.s)
        r0 = value(m.fs.steam_feed.outlet.flow_mol[0]) * Dm.MW["H2O"] \
            / (FCO2_DEF_START * Dm.KGPH)
        self.ratio = r0
        self.f = FCO2_DEF_START
        self.walk(lambda f: self._set(f, r0), FCO2_DEF_START, D_DESIGN_FCO2,
                  0.5)
        self.f = D_DESIGN_FCO2
        self.walk(lambda v: m.fs.nc_target.set_value(v),
                  value(m.fs.nc_target), D_NC, 0.25)
        self.walk(lambda v: self._set(D_DESIGN_FCO2, v), r0,
                  D_STEAM_PER_CO2, 0.1)
        self.ratio = D_STEAM_PER_CO2

    def _set(self, f, ratio):
        fs = self.m.fs
        fs.co2_feed.outlet.flow_mol[0].fix(f * Dm.KGPH / Dm.MW["CO2"])
        fs.steam_feed.outlet.flow_mol[0].fix(ratio * f * Dm.KGPH
                                             / Dm.MW["H2O"])

    def walk(self, setter, x0, x1, h0):
        m, s = self.m, self.s
        good, x, h = to_json(m, return_dict=True), x0, h0
        while abs(x1 - x) > 1e-9:
            xn = x + max(-h, min(h, x1 - x))
            setter(xn)
            r = s.solve(m, load_solutions=False)
            if r.solution:
                m.solutions.load_from(r)
            if r.solution and _converged(m, r, 1e-5):
                x, good, h = xn, to_json(m, return_dict=True), min(h0, 2 * h)
            else:
                from_json(m, sd=good)
                h /= 2
                if h < h0 / 128:
                    raise RuntimeError(f"walk failed at {x} -> {xn}")
        return x

    def to_fco2(self, f):
        self.walk(lambda v: self._set(v, self.ratio), self.f, f, 0.5)
        self.f = f

    def to_share(self, share):
        fs = self.m.fs
        self.walk(lambda v: fs.aqua_share.set_value(v),
                  value(fs.aqua_share), share, 0.1)

    def to_aqua_w(self, w):
        """Change the aqua NH3 mass fraction (fixed feed composition)."""
        fs = self.m.fs
        port = fs.aqua_feed.outlet

        def setw(v):
            x = x_mol(v)
            port.mole_frac_comp[0, "NH3"].fix(x)
            port.mole_frac_comp[0, "H2O"].fix(1 - x)
            for c in Dm.COMPS:
                if c not in ("NH3", "H2O"):
                    port.mole_frac_comp[0, c].fix(Dm.EPS)
        self.walk(setw, fs.aqua_w, w, 0.025)
        fs.aqua_w = w

    def results(self):
        m = self.m
        fs = m.fs
        st, shaft, urea, info = P.def_streams(Dm, m)
        nh3 = value(fs.nh3_feed.outlet.flow_mol[0]) * Dm.MW["NH3"] / Dm.KGPH
        if hasattr(fs, "aqua_feed"):
            aq = fs.aqua_feed.outlet
            nh3 += value(aq.flow_mol[0] * aq.mole_frac_comp[0, "NH3"]) \
                * Dm.MW["NH3"] / Dm.KGPH
        return st, shaft, urea, {
            "NH3 make-up [kg/h]": nh3,
            "urea [kg/h]": urea, "DEF [kg/h]": info["DEF [kg/h]"],
            "CO2 conversion/pass": value(fs.x_co2),
            "H/C reactor": value(fs.hc_ratio),
            "demin water [kg/h]": value(fs.water_feed.outlet.flow_mol[0])
            * Dm.MW["H2O"] / Dm.KGPH,
            "DEF NH3 [ppm]": info.get("DEF NH3 [ppm]", float("nan")),
        }


def evaluate(walker, aqua_w, s13_ratio, t_s13, w_per_nh3):
    st_d, sh_d, urea, di = walker.results()
    nh3_aq = di["NH3 make-up [kg/h]"] / B_DESIGN["recovery"]
    water = nh3_aq * w_per_nh3
    st_b, sh_b, bi = b_streams(nh3_aq, water, water * s13_ratio, t_s13,
                               aqua_w)
    e_b = elec(st_b, sh_b, 4.0)
    e_d = elec(st_d, sh_d, 4.0)
    e_bd = elec(st_b + st_d, sh_b + sh_d, 4.0)
    return {"case": "anhydrous (current)" if aqua_w is None
            else f"aqua {aqua_w:.2f}",
            **di, "B [kW]": e_b, "D [kW]": e_d,
            "B+D separate [kW]": e_b + e_d, "B+D integrated [kW]": e_bd,
            "kWh/kg urea (B+D integrated)": e_bd / urea}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fco2", type=float, default=3.82)
    ap.add_argument("--w", type=float, nargs="*", default=[0.35, 0.30, 0.25])
    a = ap.parse_args()
    A = pd.read_csv(ROOT / "surrogates" / "idaes_data" / "A.csv")
    A = A[A.get("model", "rigorous") == "rigorous"] if "model" in A else A
    s13_ratio = float((A.s13_kgph / A.water_kgph).median())
    t_s13 = float(A.t_s13.median())
    # water carried per kg aq. NH3 at the IPPS A operating point (A.csv)
    w_per_nh3 = float((A.water_kgph / A.ammonia_kgph).median())
    rows = []
    print("reference: anhydrous NH3 to D's HP pump", flush=True)
    ref = DWalker(None)
    ref.to_fco2(a.fco2)
    rows.append(evaluate(ref, None, s13_ratio, t_s13, w_per_nh3))
    print(pd.Series(rows[-1]).to_string(), flush=True)
    ref.ctx.__exit__(None, None, None)

    ws = sorted(a.w, reverse=True)
    wk = DWalker(aqua_w=ws[0])
    wk.to_fco2(a.fco2)
    try:
        wk.to_share(0.999)
    except RuntimeError as e:
        print(f"aqua share walk failed: {e}", flush=True)
    for w in ws:
        try:
            if abs(w - wk.m.fs.aqua_w) > 1e-9:
                wk.to_aqua_w(w)
        except RuntimeError as e:
            print(f"aqua w={w}: {e} (DEF water balance or convergence)",
                  flush=True)
            break
        rows.append(evaluate(wk, w, s13_ratio, t_s13, w_per_nh3))
        print(f"\n{pd.Series(rows[-1]).to_string()}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    cols = ["case", "urea [kg/h]", "CO2 conversion/pass", "H/C reactor",
            "demin water [kg/h]", "B [kW]", "D [kW]", "B+D integrated [kW]",
            "kWh/kg urea (B+D integrated)"]
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
