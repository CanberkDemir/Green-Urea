"""
End-to-end plant energy per kg urea (TODO F1, E2, E3, D5).

Chains the pieces that exist as separate IDAES models:

  electrolyzer (algebraic, paper: FE 0.98, 50 kWh/kg H2)
    -> ammonia section (ammonia_flowsheet.py, via ammonia_energy.py:
       reactor, B6 brine concentration, NH3 recovery option)
    -> DEF urea section (def_lowp_flowsheet.py, sized so its NH3 make-up
       equals the ammonia product)

and reports electricity per kg urea for each section, with and without
plant-wide heat integration (heat_integration.py), against the
manuscript's ~190 kWh/kg urea.

Electricity accounting (TODO E3) is explicit, not the paper's
sum-of-absolute-duties at COP 1:
  * shaft work as is; heating by heat pump (0.5 Carnot from 25 C) or
    resistance; cooling >= 35 C by cooling water (2% of duty as fan/pump
    power), below 35 C by the R717 chiller COP of ammonia_flowsheet.

    python plant_energy.py                      # three cases
    python plant_energy.py --def-module PATH    # use another DEF version

The DEF section runs one solve per case (~1-2 min each).
"""

import argparse
import contextlib
import importlib.util
import io
import logging
import sys
from pathlib import Path

from pyomo.environ import value

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solver_path  # noqa: E402  (Ipopt 3.14 for the DEF solve, TODO D6)
import ammonia_energy as AE  # noqa: E402
from heat_integration import Stream, pinch  # noqa: E402

E_SPEC_H2 = 50.0   # kWh/kg H2 (paper, SI)
FE = 0.98          # Faradaic efficiency (paper; Rashid et al. 2015)
MW_UREA, MW_NH3, MW_CO2 = 60.055, 17.031, 44.010


def load_def_module(path):
    if path is None:
        import def_lowp_flowsheet as D
        return D
    spec = importlib.util.spec_from_file_location("def_variant", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def def_section(D, fco2_kgph, direct=0.0, boilup=0.15):
    """Solve the DEF plant; return (streams, shaft_kw, urea_kgph, info).

    direct=0 (all condensate recycled) solves with Ipopt 3.14; the
    condensate-to-DEF option (direct>0) is still infeasible (TODO D6).
    """
    m = D.build_model(fco2_kgph=fco2_kgph, direct_frac=direct,
                      boilup=boilup)
    with contextlib.redirect_stdout(io.StringIO()), solver_path.ipopt314():
        s = D.initialize(m)
        D.solve(m, s)
    return def_streams(D, m)


def def_streams(D, m):
    """(streams, shaft_kw, urea_kgph, info) of a solved DEF model."""
    fs = m.fs
    c = lambda T: value(T) - 273.15  # noqa: E731
    streams = []
    from idaes.models.unit_models import (Flash, Heater,
                                          StoichiometricReactor)
    for u in fs.component_objects(descend_into=False):
        if not isinstance(u, (Flash, Heater, StoichiometricReactor)):
            continue
        if not hasattr(u, "heat_duty"):
            continue
        q = value(u.heat_duty[0]) / 1e3
        if abs(q) < 1e-3:
            continue
        cv = u.control_volume
        t_in, t_out = c(cv.properties_in[0].temperature), \
            c(cv.properties_out[0].temperature)
        if q > 0:   # needs heating -> cold stream
            lo, hi = min(t_in, t_out), max(t_in, t_out)
            if hi - lo < 0.1:
                hi = lo + 0.1
            streams.append(Stream(u.local_name, lo, hi, q, section="urea"))
        else:       # needs cooling -> hot stream
            hi, lo = max(t_in, t_out), min(t_in, t_out)
            if hi - lo < 0.1:
                lo = hi - 0.1
            streams.append(Stream(u.local_name, hi, lo, -q, section="urea"))
    # LP stripping steam raised from 25 C water (sensible + latent)
    n_st = value(fs.steam_feed.outlet.flow_mol[0])
    t_st = c(fs.steam_feed.outlet.temperature[0])
    streams.append(Stream("LP steam generation", 25.0, t_st,
                          n_st * (75.4 * 75 + 40.7e3 + 34 * (t_st - 100))
                          / 1e3, section="urea"))
    n_co2 = value(fs.co2_feed.outlet.flow_mol[0])
    shaft = (D.co2_compression_kw(n_co2)
             + value(fs.nh3_pump.work_mechanical[0]) / 1e3
             + value(fs.rec_pump.work_mechanical[0]) / 1e3
             + value(fs.lp_pump.work_mechanical[0]) / 1e3
             + value(fs.cond_pump.work_mechanical[0]) / 1e3
             + D.co2_compression_kw(value(fs.lp_comp.inlet.flow_mol[0]),
                                    p_in=D.P_LP, p_out=D.P_MP, stages=1,
                                    T_in=value(fs.lp_comp.inlet
                                               .temperature[0])))
    p = fs.def_prod.inlet
    urea = value(p.flow_mol[0] * p.mole_frac_comp[0, "UREA"]) * D.MW["UREA"] \
        / D.KGPH
    nh3_in = value(fs.nh3_feed.outlet.flow_mol[0]) * D.MW["NH3"] / D.KGPH
    info = {"urea [kg/h]": urea, "DEF [kg/h]": value(fs.def_mass),
            "NH3 make-up [kg/h]": nh3_in,
            "CO2 conversion/pass": value(fs.x_co2),
            "DEF NH3 [ppm]": value(p.mole_frac_comp[0, "NH3"])
            * D.MW["NH3"] / sum(value(p.mole_frac_comp[0, j]) * D.MW[j]
                                for j in D.COMPS) * 1e6}
    return streams, shaft, urea, info


def run_case(label, D, network="bkp", fh2=10.0, s16=None, tflash=None,
             recovery="M", b6_energy=True, integrate=True, dt_min=10.0):
    import reactions.ammonia_reactions as AR
    AR.NETWORK = network
    m = AE.solve_ammonia(fh2=fh2, t_flash_c=tflash, s16_water=s16)
    fs = m.fs
    nh3 = value(fs.nh3_product_kgph)
    a_streams = AE.base_streams(m)
    rs, r_shaft, _ = AE.recovery_streams(
        recovery, nh3, value(fs.h2o_product_kgph),
        value(fs.s13_mass_kgph),
        value(fs.s13.properties[0].temperature) - 273.15)
    a_streams += rs
    b6 = AE.brine_concentration(m)["B6 electricity [kW]"] if b6_energy \
        else 0.0
    a_shaft = value(fs.W_comp) / 1e3 + r_shaft + b6
    cop_ch = value(fs.refrigeration.cop)
    h2_fresh = value(fs.h2_feed.outlet.flow_mol[0]) * 2.016e-3 * 3600
    e_elec = h2_fresh * E_SPEC_H2 / FE

    # size the DEF plant on the ammonia product: NH3 : CO2 = 2 : 1
    fco2 = nh3 / MW_NH3 / 2 * MW_CO2
    u_streams, u_shaft, urea, u_info = def_section(D, fco2)

    def elec(streams, shaft):
        res = pinch(streams, dt_min)
        return AE.electricity(res, streams, shaft, cop_ch,
                              integrated=integrate)[
            "TOTAL, heat-pump heating [kWe]"], res

    a_kw, _ = elec(a_streams, a_shaft)
    u_kw, _ = elec(u_streams, u_shaft)
    plant_kw, res_all = elec(a_streams + u_streams, a_shaft + u_shaft)
    out = {
        "NH3 [kg/h]": nh3, "fresh H2 [kg/h]": h2_fresh,
        "urea [kg/h]": urea, "DEF [kg/h]": u_info["DEF [kg/h]"],
        "electrolysis [kWh/kg urea]": e_elec / urea,
        "ammonia section [kWh/kg urea]": a_kw / urea,
        "urea/DEF section [kWh/kg urea]": u_kw / urea,
        "TOTAL, sections separate [kWh/kg urea]":
            (e_elec + a_kw + u_kw) / urea,
        "TOTAL, plant-wide HI [kWh/kg urea]": (e_elec + plant_kw) / urea,
        "plant-wide HI saving [kW]": a_kw + u_kw - plant_kw,
    }
    print(f"\n=== {label} ===")
    for k, v in out.items():
        print(f"  {k:42s} {v:10.2f}")
    print("  " + res_all.summary())
    return out


# ------------------------------------------- direct NO3RR route (electro)
T_CELL = 65.0      # C, cell / catholyte loop temperature (assumed)
V_TN_NO3RR = 0.607 # V, thermoneutral NO3- + 2H2O -> NH3 + 2O2 + OH- (aq dHf)
V_TN_HER = 1.481   # V, thermoneutral water splitting
COP_CHILL_ELECTRO = 4.0  # no R717 loop here; generic 10 C chiller


def _flip(streams):
    for s_ in streams:
        if s_.duty_kw < 0:
            s_.t_in, s_.t_out, s_.duty_kw = s_.t_out, s_.t_in, -s_.duty_kw
    return [s_ for s_ in streams if s_.duty_kw > 1e-6]


def run_electro_case(label, D, params="central", stages=4, conversion=0.95,
                     recovery="S2", integrate=True, dt_min=10.0):
    """Direct NO3RR cascade (electro_no3rr.py) on raw brine -> NH3 recovery
    (ammonia_energy.recovery_streams) -> DEF section. No water electrolysis,
    H2 compression, hydrogenation reactor or B6."""
    import electro_no3rr as EN
    _, t = EN.design(EN.CASES[params], "raw", stages, conversion)
    nh3, cell_kw = t["nh3_kgph"], t["power_kw"]
    i_n = 8 * EN.F * EN.feed("raw")[0] * conversion            # A to NH3
    i_h = t["current"] - i_n                                   # A to HER
    v_avg = cell_kw * 1e3 / t["current"]
    q_cell = (i_n * (v_avg - V_TN_NO3RR) + i_h * (v_avg - V_TN_HER)) / 1e3
    h2o = EN.BRINE_KGPH["H2O"]
    mass = sum(EN.BRINE_KGPH.values())
    a_streams = [
        Stream("cell feed preheat", 25.0, T_CELL - 2.0,
               mass * AE.CP_BRINE * (T_CELL - 27.0) / 3600, section="electro"),
        Stream("cell heat rejection", T_CELL, T_CELL - 0.1, q_cell,
               section="electro"),
    ]
    rs, r_shaft, r_info = AE.recovery_streams(recovery, nh3, h2o, mass,
                                              T_CELL)
    a_streams = _flip(a_streams + rs)
    a_shaft = cell_kw + r_shaft

    fco2 = nh3 / MW_NH3 / 2 * MW_CO2
    u_streams, u_shaft, urea, u_info = def_section(D, fco2)

    def elec(streams, shaft):
        res = pinch(streams, dt_min)
        return AE.electricity(res, streams, shaft, COP_CHILL_ELECTRO,
                              integrated=integrate)[
            "TOTAL, heat-pump heating [kWe]"], res

    a_kw, a_res = elec(a_streams, a_shaft)
    u_kw, _ = elec(u_streams, u_shaft)
    plant_kw, res_all = elec(a_streams + u_streams, a_shaft + u_shaft)
    out = {
        "NH3 [kg/h]": nh3, "cell power [kW]": cell_kw,
        "cell FE": t["fe"], "cell waste heat [kW]": q_cell,
        "by-product H2 (not credited) [kg/h]": t["h2_kgph"],
        "stripping vapour [kg/h]": r_info.get("stripping vapour [kg/h]", 0.0),
        "ammonia section [kWh/kg NH3]": a_kw / nh3,
        "urea [kg/h]": urea, "DEF [kg/h]": u_info["DEF [kg/h]"],
        "electrolysis [kWh/kg urea]": 0.0,
        "ammonia section [kWh/kg urea]": a_kw / urea,
        "urea/DEF section [kWh/kg urea]": u_kw / urea,
        "TOTAL, sections separate [kWh/kg urea]": (a_kw + u_kw) / urea,
        "TOTAL, plant-wide HI [kWh/kg urea]": plant_kw / urea,
        "plant-wide HI saving [kW]": a_kw + u_kw - plant_kw,
    }
    print(f"\n=== {label} ===")
    for k, v in out.items():
        print(f"  {k:42s} {v:10.2f}")
    print("  ammonia: " + a_res.summary())
    print("  plant:   " + res_all.summary())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--def-module", default=None,
                    help="path to a def_lowp_flowsheet variant")
    ap.add_argument("--electro-only", action="store_true",
                    help="only run the direct-NO3RR cases")
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    D = load_def_module(a.def_module)

    cases = [
        ("1 paper-like: bkp kinetics, B6 to 1400 kg/h, membrane, 10 C "
         "flash, no heat integration",
         dict(network="bkp", fh2=10.0, s16=1400, tflash=10.0,
              recovery="M", integrate=False)),
        ("2 improved: bkp kinetics, B6 to 5000 kg/h, vacuum stripper, "
         "50 C flash, heat integration",
         dict(network="bkp", fh2=10.0, s16=5000, tflash=50.0,
              recovery="S2", integrate=True)),
        ("3 improved + high-selectivity catalyst (SI network, Ru-like)",
         dict(network="si", fh2=11.5, s16=5000, tflash=50.0,
              recovery="S2", integrate=True)),
    ]
    electro = [
        (f"4{c[0]} direct NO3RR ({c}), 4-stage cascade, raw brine, vacuum "
         "stripper, heat integration", dict(params=c))
        for c in ("optimistic", "central", "pessimistic")]
    results = {}
    for label, kw in ([] if a.electro_only else cases):
        try:
            results[label] = run_case(label, D, **kw)
        except Exception as e:  # keep going; report the failure
            print(f"\n=== {label} ===\n  FAILED: {str(e)[:150]}")
    for label, kw in electro:
        try:
            results[label] = run_electro_case(label, D, **kw)
        except Exception as e:
            print(f"\n=== {label} ===\n  FAILED: {str(e)[:150]}")
    print("\nSUMMARY (kWh electricity per kg urea; paper reports ~190)")
    for label, r in results.items():
        print(f"  {label.split(':')[0]:4s} electrolysis "
              f"{r['electrolysis [kWh/kg urea]']:6.1f} | ammonia "
              f"{r['ammonia section [kWh/kg urea]']:5.1f} | urea "
              f"{r['urea/DEF section [kWh/kg urea]']:5.1f} | total "
              f"{r['TOTAL, sections separate [kWh/kg urea]']:6.1f} | "
              f"plant-wide HI {r['TOTAL, plant-wide HI [kWh/kg urea]']:6.1f}")


if __name__ == "__main__":
    main()
