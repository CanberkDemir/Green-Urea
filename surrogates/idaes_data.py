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
    if f0 < 9.5 or target < 9.5:
        return _walk_h2_adaptive(m, solver, f0, target, step)
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
            try:
                ok = bool(r.solution) and _converged(m, r)
            except ValueError:      # residual not evaluable (log of <0)
                ok = False
            if ok:
                break
        else:
            if last:
                raise RuntimeError(f"H2 walk failed at {f:.2f} kg/h")


def _walk_h2_adaptive(m, solver, f0, target, step, min_step=0.005):
    """H2-starved region: steps of at most `step`, halved on failure
    from the last converged state (part of walk_h2)."""
    from ammonia_flowsheet import MW
    from idaes.core.util.model_serializer import from_json, to_json
    port = m.fs.h2_feed.outlet
    f, h = f0, min(step, 0.05)
    import ammonia_flowsheet as A
    # fallback: adaptive barrier, no bound relaxation (keeps the PFR's
    # H2 >= 0 after the depletion front; converges where the default fails)
    alt = A.get_solver("ipopt", options={
        "tol": 1e-6, "max_iter": 1000, "mu_strategy": "adaptive",
        "bound_push": 1e-10, "bound_relax_factor": 0.0})
    good = to_json(m, return_dict=True)
    nh3 = value(m.fs.nh3_product_kgph)
    while abs(target - f) > 1e-9:
        g = target if abs(target - f) <= h else f + h * np.sign(target - f)
        ok = False
        for sv in (solver, alt):
            from_json(m, sd=good)
            port.flow_mol[0].fix(g / 3600.0 / MW["H2"])
            try:
                r = sv.solve(m, load_solutions=False)
                ok = bool(r.solution)
                if ok:
                    m.solutions.load_from(r)
                    ok = _converged(m, r)
            except Exception:
                ok = False
            # reject a jump to another steady-state branch (NH3 is
            # ~1.22*Fh2 here: a small Fh2 step moves it by ~1.22*dFh2)
            if ok and abs(value(m.fs.nh3_product_kgph) - nh3) > \
                    0.5 + 3.0 * abs(g - f):
                ok = False
            if ok:
                break
        if ok:
            nh3 = value(m.fs.nh3_product_kgph)
            f = g
            good = to_json(m, return_dict=True)
            h = min(h * 1.5, step)
        else:
            from_json(m, sd=good)
            h /= 2.0
            if h < min_step:
                raise RuntimeError(f"H2 walk failed at {g:.3f} kg/h "
                                   f"(last good {f:.3f})")


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
                    if fh2 < 9.0 and abs(value(fs.h2_feed.outlet.flow_mol[0])
                                         * 3600 * A.MW["H2"] - fh2) > 1e-3:
                        raise RuntimeError("did not reach target")
                except RuntimeError as e:
                    print(f"  A Ft={ft:.0f} Fh2={fh2:.1f}: {e}", flush=True)
                    from_json(m, sd=good)
                    continue
                prev, good = good, to_json(m, return_dict=True)
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
                bad = [k for k, v in r.items() if not np.isfinite(v)
                       or (k != "t_s13" and v < -1e-6)]
                if r["ammonia_kgph"] < 0.5 * 1.22 * r["Fh2"]:
                    bad.append("ammonia_kgph<0.61*Fh2")
                if bad:
                    print(f"  A Ft={ft:.0f} Fh2={fh2:5.2f} REJECTED "
                          f"NH3={r['ammonia_kgph']:.3f} {bad}", flush=True)
                    good = prev
                    from_json(m, sd=good)
                    continue
                rows.append(r)
                print(f"  A Ft={ft:.0f} Fh2={fh2:5.2f} NH3={r['ammonia_kgph']:6.2f}"
                      f" E={e:7.1f} kW ({time.time() - t0:.0f}s)", flush=True)
    ctx314.__exit__(None, None, None)
    new = pd.DataFrame(rows)
    if len(new):
        new["model"] = "rigorous"
    df = merge(new, "A", ["Ft", "Fh2"])
    print(f"wrote {OUT / 'A.csv'} ({len(df)} points)")


def section_a_h2_limited(ft_grid, fh2_grid, knee_ratio=1.215):
    """Reduced A model for the H2-starved regime (fresh H2 fully consumed).
    Selectivity NH3/Fh2 and the linear responses of the other outputs are
    taken from rigorous points in that regime (NH3/Fh2 >= knee_ratio);
    B6 power is the exact formula. Only fills grid points that have no
    rigorous row, and only where Fh2 lies below the nitrate-limit knee."""
    import ammonia_flowsheet as A
    f = OUT / "A.csv"
    d = pd.read_csv(f)
    if "model" not in d:
        d["model"] = "rigorous"
    d["model"] = d["model"].fillna("rigorous")
    rig = d[d.model == "rigorous"]
    lim = rig[rig.ammonia_kgph / rig.Fh2 >= knee_ratio]
    sel = float(np.median(lim.ammonia_kgph / lim.Fh2))
    X = np.c_[np.ones(len(lim)), lim.Ft, lim.Fh2]
    fit = {k: np.linalg.lstsq(X, lim[k], rcond=None)[0]
           for k in ("water_kgph", "s13_kgph", "t_s13", "electric_kwhph")}
    kc = float(np.median(rig.h2_comp_kw / rig.Fh2))
    # nitrate-limit plateau: NH3_max(Ft) from the rigorous plateau points
    plat = rig[rig.ammonia_kgph / rig.Fh2 < knee_ratio - 0.01]
    nh3max = np.polyfit(plat.Ft, plat.ammonia_kgph, 1) if len(plat) > 1 \
        else None
    have = set(zip(rig.Ft.round(3), rig.Fh2.round(3)))
    rows = []
    for ft in ft_grid:
        for fh2 in fh2_grid:
            if (round(ft, 3), round(fh2, 3)) in have:
                continue
            nh3 = sel * fh2
            if nh3max is not None and nh3 > 0.995 * np.polyval(nh3max, ft):
                continue          # past the knee: not H2-limited
            x = np.array([1.0, ft, fh2])
            w_in = A.BRINE_KGPH["H2O"] * ft / 8500.0
            rows.append(dict(
                Ft=ft, Fh2=fh2, ammonia_kgph=nh3,
                water_kgph=x @ fit["water_kgph"],
                s13_kgph=x @ fit["s13_kgph"], t_s13=x @ fit["t_s13"],
                h2_comp_kw=kc * fh2,
                b6_kw=max(w_in - S16_WATER, 0.0) / 1000.0 * AE.E_MVC,
                electric_kwhph=x @ fit["electric_kwhph"],
                model="h2_limited"))
    new = pd.DataFrame(rows)
    print(f"h2_limited: selectivity {sel:.5f} kg NH3/kg H2 from "
          f"{len(lim)} rigorous points, {len(new)} points added")
    df = merge(pd.concat([d, new], ignore_index=True), "A", ["Ft", "Fh2"])
    print(f"wrote {f} ({len(df)} points)")
    return sel, fit


# ----------------------------------------------------------------- B
B_NH3_RANGE = (1.0, 14.0)       # kg/h, covers A.csv (9.7-12.8) and low flows
B_WATER_RANGE = (2000.0, 5500.0)  # kg/h
B_N = 40                          # grid points per axis
# optimised S2 design (idaes_model/recovery_opt.py, at the IPPS point
# NH3 ~9.85 kg/h, water ~5056 kg/h): vapour-recompression stripper at
# 0.06 bar (t_b = Tsat + 5.5 K BPE = 41.7 C), 20 stages, R = 1.2 Rmin,
# rectifier feed preheated to 180 C, aqua condensate cooled to 25 C
B_DESIGN = dict(p_strip=0.06, n_stages=20, recovery=0.99, r_factor=1.2,
                t_rect_feed=180.0, t_ov_cond=25.0, overhead_mvr=True)


def section_b(n=B_N, nh3_range=B_NH3_RANGE, water_range=B_WATER_RANGE):
    a = pd.read_csv(OUT / "A.csv")
    ratio = float((a.s13_kgph / a.water_kgph).median())
    t_s13 = float(a.t_s13.median())
    rows, bad = [], []
    for nh3 in np.linspace(*nh3_range, n):
        for w in np.linspace(*water_range, n):
            try:
                st, shaft, info = AE.recovery_streams("S2", nh3, w,
                                                      w * ratio, t_s13,
                                                      **B_DESIGN)
                e = elec(st, shaft, 4.0)
                x_f = info["overhead NH3 [mol%]"] / 100
                if not (0 < x_f < AE.X_D) or info["rectifier R (=1.5 Rmin)"]                         <= B_DESIGN["r_factor"] * 0.1 + 1e-12:
                    raise ValueError(f"Underwood invalid x_f={x_f:.3g}")
            except Exception as ex:  # shortcut model breakdown
                bad.append((nh3, w, repr(ex)))
                continue
            if not np.isfinite(e) or e <= 0:
                bad.append((nh3, w, f"E={e}"))
                continue
            rows.append(dict(ammonia_kgph=nh3, water_kgph=w,
                             anhydrous_nh3_kgph=B_DESIGN["recovery"] * nh3,
                             electric_kwhph=e))
    if bad:
        print(f"B: dropped {len(bad)} invalid points, e.g. {bad[:3]}")
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
        # build at the default design (cold start does not converge at low
        # steam), then walk steam/CO2 and N/C to the optimized D design
        m = Dm.build_model(fco2_kgph=FCO2_DEF_START, direct_frac=0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            s = Dm.initialize(m)
            Dm.solve(m, s)
        # the steam cut does not converge directly at Fco2 20; do it at
        # D_DESIGN_FCO2: walk down (1 kg steam/kg CO2), change N/C and the
        # steam ratio there, then walk back up at the optimized ratio
        def _set(f, ratio):
            m.fs.co2_feed.outlet.flow_mol[0].fix(f * Dm.KGPH / Dm.MW["CO2"])
            m.fs.steam_feed.outlet.flow_mol[0].fix(
                ratio * f * Dm.KGPH / Dm.MW["H2O"])

        def _walk(setter, x0, x1, h0):
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
                    if h < h0 / 64:
                        raise RuntimeError(f"D design walk failed at {x}")

        r0 = value(m.fs.steam_feed.outlet.flow_mol[0]) * Dm.MW["H2O"] \
            / (FCO2_DEF_START * Dm.KGPH)
        _walk(lambda f: _set(f, r0), FCO2_DEF_START, D_DESIGN_FCO2, 0.5)
        _walk(lambda v: m.fs.nc_target.set_value(v),
              value(m.fs.nc_target), D_NC, 0.25)
        _walk(lambda v: _set(D_DESIGN_FCO2, v), r0, D_STEAM_PER_CO2, 0.1)
        _walk(lambda f: _set(f, D_STEAM_PER_CO2), D_DESIGN_FCO2,
              FCO2_DEF_START, 0.5)
        port = m.fs.co2_feed.outlet
        n0 = value(port.flow_mol[0])
        # stripper steam scaled with Fco2 (fixed 20 kg/h over-dilutes the
        # DEF below ~8.5 kg/h CO2 and the water make-up hits its lb of 0)
        steam = m.fs.steam_feed.outlet.flow_mol[0]
        s0 = value(steam)
        for grid in (sorted([f for f in FCO2_DEF if f >= FCO2_DEF_START]),
                     sorted([f for f in FCO2_DEF if f < FCO2_DEF_START],
                            reverse=True)):
            port.flow_mol[0].fix(n0)
            steam.fix(s0)
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
                    steam.fix(s0 * f / FCO2_DEF_START)
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
# optimized D design at the IPPS point (Fco2 3.63 kg/h, sweep of steam/CO2
# 0.2-1.5 x N/C 3.5-5): stripping steam heated from 25 C and recycled as
# condensate water to the 190 C reactor dominates; 0.2 kg steam/kg CO2 and
# N/C 3.5 (patent lower bound) cut E from 0.90 to 0.52 kWh/kg urea.
# Defaults in def_lowp_flowsheet (20 kg/h steam, N/C 4) are unchanged.
D_STEAM_PER_CO2 = 0.2   # kg LP stripping steam / kg CO2 (was 1.0)
D_NC = 3.5              # reactor inlet N/C (was 4.0)
D_DESIGN_FCO2 = 6.0     # Fco2 at which the design change is walked in


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
