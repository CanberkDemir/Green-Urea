"""
IDAES training data for the IPPS surrogates, one table per plant section
(TODO F4). Replaces the Aspen grids in aspen/data for the IPPS model.

  A  ammonia section (idaes_model/ammonia_flowsheet.py, plant_energy case 2:
     .bkp kinetics, B6 to 5000 kg/h water, 50 C flash)
     inputs  Ft [kg/h brine], Fh2 [kg/h]
     outputs ammonia_kgph, water_kgph (aqueous NH3 product S13),
             electric_kwhph (H2 compression + B6 MVC + utilities after
             section pinch; the water electrolyser is NOT included, IPPS
             models it separately as P_el)
  B  NH3 recovery (ammonia_energy.recovery_streams, option S2: vacuum
     stripper + 15 bar rectifier, shortcut)
     inputs  ammonia_kgph, water_kgph (the A outputs)
     outputs anhydrous_nh3_kgph, electric_kwhph
  C  urea section (idaes_model/urea_flowsheet.py, ureaF replica)
     inputs  Fnh3, Fco2 [kg/h]
     outputs pure_urea_kgph, product_urea_wtfrac, electric_kwhph

Electricity everywhere: ammonia_energy.electricity(..., integrated=True),
i.e. pinch per section at dT_min 10 K, heat pump (0.5 Carnot) for hot
utility, cooling water / chiller for cold utility, plus shaft work.

    python idaes_data.py A        # ~10 min, continuation walk
    python idaes_data.py B        # seconds (needs A)
    python idaes_data.py C        # ~1 min per point
Writes surrogates/idaes_data/{A,B,C}.csv.
"""

import contextlib
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "idaes_model"))
OUT = ROOT / "surrogates" / "idaes_data"
OUT.mkdir(exist_ok=True)
logging.disable(logging.WARNING)

import ammonia_energy as AE  # noqa: E402
from heat_integration import Stream, pinch  # noqa: E402
from pyomo.environ import value  # noqa: E402

DT_MIN = 10.0
S16_WATER = 5000.0
T_FLASH = 50.0
FT_GRID = np.linspace(8500.0, 9080.0, 5)
FH2_GRID = np.linspace(5.0, 15.0, 11)
FNH3_GRID = np.linspace(5.0, 12.0, 6)
FCO2_GRID = np.linspace(3.0, 50.0, 7)


def elec(streams, shaft_kw, cop_chiller):
    streams = [s for s in streams if s.duty_kw > 1e-6]
    res = pinch(streams, DT_MIN)
    return AE.electricity(res, streams, shaft_kw, cop_chiller,
                          integrated=True)["TOTAL, heat-pump heating [kWe]"]


# ----------------------------------------------------------------- A
def _converged(m, r, tol=1e-5):
    """Ipopt converges but sometimes prints 'Error in eq_mult_calculator'
    (square problem, zero objective), which Pyomo maps to
    internalSolverError. Accept the point if the residual is small."""
    if str(r.solver.termination_condition) == "optimal":
        return True
    from pyomo.environ import Constraint
    res = max((abs(value(c.body) - value(c.upper if c.upper is not None
                                          else c.lower))
               for c in m.component_data_objects(Constraint, active=True)
               if c.equality), default=0.0)
    return res < tol


def walk_h2(m, solver, target, step=0.25):
    from ammonia_flowsheet import MW
    port = m.fs.h2_feed.outlet
    f0 = value(port.flow_mol[0]) * 3600.0 * MW["H2"]
    n = max(1, int(abs(target - f0) / step + 0.999))
    for i in range(1, n + 1):
        f = f0 + (target - f0) * i / n
        last = i == n
        # an isolated step sometimes returns no solution; the next one
        # converges, so intermediate failures are skipped and the target
        # is nudged by +-0.05 % if needed
        for g in ((f, f * 1.0005, f * 0.9995) if last else (f,)):
            port.flow_mol[0].fix(g / 3600.0 / MW["H2"])
            r = solver.solve(m, load_solutions=False)
            if r.solution:
                m.solutions.load_from(r)
            if r.solution and _converged(m, r):
                break
        else:
            if last:
                raise RuntimeError(f"H2 walk failed at {f:.2f} kg/h")


def section_a():
    import ammonia_flowsheet as A
    import reactions.ammonia_reactions as AR
    AR.NETWORK = "bkp"
    with contextlib.redirect_stdout(io.StringIO()):
        m = AE.solve_ammonia(fh2=10.0, t_flash_c=T_FLASH,
                             s16_water=S16_WATER)
    fs = m.fs
    solver = A.get_solver("ipopt", options={"tol": 1e-6, "max_iter": 500})
    n_brine0 = value(fs.no3_feed.outlet.flow_mol[0])
    w_in0 = A.BRINE_KGPH["H2O"]
    rows = []
    # start at Fh2 = 10 and walk outwards in both directions per Ft
    fh2_up = sorted(f for f in FH2_GRID if f >= 10.0)
    fh2_dn = sorted((f for f in FH2_GRID if f < 10.0), reverse=True)
    from idaes.core.util.model_serializer import from_json, to_json
    import solver_path
    # the base point needs Ipopt 3.13 (B6 init), but the continuation
    # steps away from it converge far better on 3.14 (Ft below 8500 fails
    # on 3.13 at the first 50 kg/h step, converges on 3.14)
    ctx314 = solver_path.ipopt314()   # keep a reference: the generator
    ctx314.__enter__()               # context restores 3.13 when collected
    base = to_json(m, return_dict=True)
    snap_ft = base
    for ft in FT_GRID:
        # walk Ft from the previous Ft's Fh2 = 10 state
        from_json(m, sd=snap_ft)
        try:
            for k in range(1, 4):
                n0 = value(fs.no3_feed.outlet.flow_mol[0])
                fs.no3_feed.outlet.flow_mol[0].fix(
                    n0 + (n_brine0 * ft / 8500.0 - n0) * k / 3)
                walk_h2(m, solver, 10.0)
        except RuntimeError as e:
            print(f"  A Ft={ft:.0f}: {e}")
            continue
        snap_ft = to_json(m, return_dict=True)
        for branch in (fh2_up, fh2_dn):
            from_json(m, sd=snap_ft)
            good = snap_ft
            for fh2 in branch:
                t0 = time.time()
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        walk_h2(m, solver, fh2, 0.25 if fh2 >= 10 else 0.1)
                except RuntimeError as e:
                    print(f"  A Ft={ft:.0f} Fh2={fh2:.1f}: {e}", flush=True)
                    from_json(m, sd=good)
                    continue
                good = to_json(m, return_dict=True)
                st = AE.base_streams(m)
                w_in = w_in0 * ft / 8500.0
                b6 = max(w_in - S16_WATER, 0.0) / 1000.0 * AE.E_MVC
                shaft = value(fs.W_comp) / 1e3 + b6
                e = elec(st, shaft, value(fs.refrigeration.cop))
                r = dict(Ft=ft, Fh2=value(fs.h2_feed.outlet.flow_mol[0]) * 3600 * A.MW["H2"],
                         ammonia_kgph=value(fs.nh3_product_kgph),
                         water_kgph=value(fs.h2o_product_kgph),
                         s13_kgph=value(fs.s13_mass_kgph),
                         t_s13=value(fs.s13.properties[0].temperature)
                         - 273.15,
                         h2_comp_kw=value(fs.W_comp) / 1e3, b6_kw=b6,
                         electric_kwhph=e)
                rows.append(r)
                print(f"  A Ft={ft:.0f} Fh2={fh2:5.2f} NH3={r['ammonia_kgph']:6.2f}"
                      f" E={e:7.1f} kW ({time.time() - t0:.0f}s)", flush=True)
    ctx314.__exit__(None, None, None)
    df = merge(pd.DataFrame(rows), "A", ["Ft", "Fh2"])
    print(f"wrote {OUT / 'A.csv'} ({len(df)} points)")


# ----------------------------------------------------------------- B
def section_b(n=15):
    a = pd.read_csv(OUT / "A.csv")
    ratio = float((a.s13_kgph / a.water_kgph).median())
    t_s13 = float(a.t_s13.median())
    rows = []
    for nh3 in np.linspace(a.ammonia_kgph.min(), a.ammonia_kgph.max(), n):
        for w in np.linspace(a.water_kgph.min(), a.water_kgph.max(), n):
            st, shaft, info = AE.recovery_streams("S2", nh3, w, w * ratio,
                                                  t_s13)
            e = elec(st, shaft, 4.0)
            rows.append(dict(ammonia_kgph=nh3, water_kgph=w,
                             anhydrous_nh3_kgph=AE.RECOVERY * nh3,
                             electric_kwhph=e))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "B.csv", index=False)
    print(f"wrote {OUT / 'B.csv'} ({len(df)} points); "
          f"E {df.electric_kwhph.min():.1f}-{df.electric_kwhph.max():.1f} kW")


# ----------------------------------------------------------------- C
def urea_streams(m):
    """Heat-duty streams and shaft work of every unit in the urea replica."""
    fs = m.fs
    st, shaft = [], 0.0
    for u in fs.component_objects(descend_into=False):
        if hasattr(u, "work_mechanical"):
            shaft += max(value(u.work_mechanical[0]), 0.0) / 1e3
            continue
        if not hasattr(u, "heat_duty") or not hasattr(u, "control_volume"):
            continue
        try:
            q = value(u.heat_duty[0]) / 1e3
            cv = u.control_volume
            ti = value(cv.properties_in[0].temperature) - 273.15
            to = value(cv.properties_out[0].temperature) - 273.15
        except Exception:
            continue
        if abs(q) < 1e-3:
            continue
        lo, hi = min(ti, to), max(ti, to)
        if hi - lo < 0.1:
            hi = lo + 0.1
        if q > 0:
            st.append(Stream(u.local_name, lo, hi, q, section="urea"))
        else:
            st.append(Stream(u.local_name, hi, lo, -q, section="urea"))
    return st, shaft


def section_c():
    """Continuation from the converged base point (Fnh3 10, Fco2 15): cold
    starts away from it rarely converge. Walk Fnh3 in 0.5 kg/h steps from
    the base, then Fco2 in 2.5 kg/h steps along each Fnh3 line."""
    import urea_flowsheet as U
    from idaes.core.util.model_serializer import from_json, to_json
    with contextlib.redirect_stdout(io.StringIO()):
        m = U.build_model(fnh3_kgph=10.0, fco2_kgph=15.0)
        s = U.initialize(m)
        r = U.solve(m, s, tee=False)
    assert str(r.solver.termination_condition) == "optimal"
    fs = m.fs
    MWk = {"NH3": 17.031e-3, "CO2": 44.01e-3}
    n_port, c_port = fs.nh3_feed.outlet, fs.co2_feed.outlet

    def kgph(port, sp):
        return value(port.flow_mol[0]) * 3600 * MWk[sp]

    def step(fnh3, fco2):
        n_port.flow_mol[0].fix(fnh3 / 3600 / MWk["NH3"])
        c_port.flow_mol[0].fix(fco2 / 3600 / MWk["CO2"])
        for _ in range(3):
            with contextlib.redirect_stdout(io.StringIO()):
                r = s.solve(m, load_solutions=False)
            if r.solution:
                m.solutions.load_from(r)
            if r.solution and _converged(m, r, 1e-4):
                return True
        return False

    def walk(target_nh3, target_co2, dn=0.5, dc=2.5):
        f0n, f0c = kgph(n_port, "NH3"), kgph(c_port, "CO2")
        k = max(1, int(max(abs(target_nh3 - f0n) / dn,
                           abs(target_co2 - f0c) / dc) + 0.999))
        for i in range(1, k + 1):
            if not step(f0n + (target_nh3 - f0n) * i / k,
                        f0c + (target_co2 - f0c) * i / k):
                return False
        return True

    def record(fnh3, fco2):
        st, shaft = urea_streams(m)
        e = elec(st, shaft, 4.0)
        urea = value(fs.ft_def_out) * value(fs.w_urea)
        rows.append(dict(Fnh3=fnh3, Fco2=fco2, pure_urea_kgph=urea,
                         product_urea_wtfrac=value(fs.w_urea),
                         electric_kwhph=e))
        print(f"  C Fnh3={fnh3:.1f} Fco2={fco2:.1f} urea={urea:.2f} "
              f"w={value(fs.w_urea):.3f} E={e:.1f} kW", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "C.csv", index=False)

    rows = []
    base = to_json(m, return_dict=True)
    line = base
    nh3_order = [10.0, 10.5, 11.0, 11.5, 12.0]
    nh3_down = [9.5, 9.0, 8.5, 8.0]
    for order in (nh3_order, nh3_down):
        line = base
        for fnh3 in order:
            from_json(m, sd=line)
            if not walk(fnh3, 15.0):
                print(f"  C Fnh3={fnh3:.1f}: walk failed", flush=True)
                break
            line = to_json(m, return_dict=True)
            for fco2s in (FCO2_UP, FCO2_DN):
                from_json(m, sd=line)
                for fco2 in fco2s:
                    if fco2 != 15.0 and not walk(fnh3, fco2):
                        print(f"  C Fnh3={fnh3:.1f} Fco2={fco2:.1f}: failed",
                              flush=True)
                        break
                    record(fnh3, fco2)
    print(f"wrote {OUT / 'C.csv'} ({len(rows)} points)")


FCO2_UP = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
FCO2_DN = [12.5, 10.0]


# ----------------------------------------------------------------- D
def section_d():
    """Low-pressure DEF plant (def_lowp_flowsheet.py, all condensate
    recycled, Ipopt 3.14). The urea route of the IDAES plant; used by the
    IPPS in place of C because the ureaF replica cannot reach the IPPS
    urea spec (see docs). Total recycle fixes the NH3 make-up, so the only
    input is Fco2."""
    import plant_energy as P
    import solver_path
    from idaes.core.util.model_serializer import from_json, to_json
    Dm = P.load_def_module(None)
    rows = []

    def record(m, fco2):
        st, shaft, urea, info = P.def_streams(Dm, m)
        e = elec(st, shaft, 4.0)
        w = urea / info["DEF [kg/h]"]
        rows.append(dict(Fco2=fco2, pure_urea_kgph=urea,
                         nh3_makeup_kgph=info["NH3 make-up [kg/h]"],
                         product_urea_wtfrac=w, electric_kwhph=e))
        print(f"  D Fco2={fco2:.2f} urea={urea:.2f} NH3={rows[-1]['nh3_makeup_kgph']:.2f}"
              f" w={w:.3f} E={e:.1f} kW", flush=True)
        merge(pd.DataFrame(rows), "D", ["Fco2"])

    with solver_path.ipopt314():
        m = Dm.build_model(fco2_kgph=FCO2_DEF_START, direct_frac=0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            s = Dm.initialize(m)
            Dm.solve(m, s)
        port = m.fs.co2_feed.outlet
        n0 = value(port.flow_mol[0])
        for grid in (sorted([f for f in FCO2_DEF if f >= FCO2_DEF_START]),
                     sorted([f for f in FCO2_DEF if f < FCO2_DEF_START],
                            reverse=True)):
            port.flow_mol[0].fix(n0)
            with contextlib.redirect_stdout(io.StringIO()):
                s.solve(m)
            cur = FCO2_DEF_START
            for fco2 in grid:
                # adaptive step: halve on failure (down to D_STEP / 16)
                ok, f_ok, h = True, cur, D_STEP
                good = to_json(m, return_dict=True)
                while abs(fco2 - f_ok) > 1e-9:
                    f = f_ok + max(-h, min(h, fco2 - f_ok))
                    port.flow_mol[0].fix(n0 * f / FCO2_DEF_START)
                    r = s.solve(m, load_solutions=False)
                    if r.solution:
                        m.solutions.load_from(r)
                    if r.solution and _converged(m, r, 1e-5):
                        f_ok, good = f, to_json(m, return_dict=True)
                        h = min(D_STEP, 2 * h)
                    else:
                        from_json(m, sd=good)
                        h /= 2
                        if h < D_STEP / 16:
                            ok = False
                            break
                if not ok:
                    print(f"  D Fco2={fco2:.2f}: walk failed", flush=True)
                    break
                cur = fco2
                record(m, fco2)
    print(f"wrote {OUT / 'D.csv'} ({len(rows)} points)")


FCO2_DEF_START = 20.0
D_STEP = 0.5


def merge(new, sec, keys):
    """Add new points to an existing table (domain expansion)."""
    f = OUT / f"{sec}.csv"
    if f.exists() and len(new):
        new = pd.concat([pd.read_csv(f), new], ignore_index=True)
    if len(new):
        new = new.round({k: 3 for k in keys}).drop_duplicates(
            keys, keep="last").sort_values(keys)
        new.to_csv(f, index=False)
    return new
FCO2_DEF = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]


if __name__ == "__main__":
    # domain expansion: python idaes_data.py A --ft 8000 7500 --fh2 8 7
    #                   python idaes_data.py D --fco2 9 8 7 --step 0.1
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("section")
    ap.add_argument("--ft", type=float, nargs="*")
    ap.add_argument("--fh2", type=float, nargs="*")
    ap.add_argument("--fco2", type=float, nargs="*")
    ap.add_argument("--step", type=float)
    a = ap.parse_args()
    if a.ft:
        FT_GRID = a.ft
    if a.fh2:
        FH2_GRID = a.fh2
    if a.fco2:
        FCO2_DEF = a.fco2
    if a.step:
        D_STEP = a.step
    {"A": section_a, "B": section_b, "C": section_c, "D": section_d}[a.section]()
