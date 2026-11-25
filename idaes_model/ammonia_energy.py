"""
Ammonia-side energy breakdown, NH3-recovery options and heat integration.

1. Solves the IDAES ammonia flowsheet (ammonia_flowsheet.py) and lists
   every heating / cooling / shaft duty with its temperatures.
2. Adds the missing hydrous -> anhydrous NH3 recovery step (TODO B1) as
   one of several options:
     M   gas-stripping membrane, paper's black box: 7.1 kWh/kg N
         (Iddya et al. 2020), electricity, no temperature information;
     S1  atmospheric steam stripper (100 C) + rectifier (15 bar);
     S2  vacuum stripper (0.123 bar, 50 C, so the 60 C reactor heat can
         drive its reboiler) + rectifier (15 bar);
     S3  S1 with bottoms mechanical vapour recompression (MVR) supplying
         the stripping steam.
   The column options are SHORTCUT designs (Kremser stripping section,
   Underwood minimum reflux), not rigorous columns - see TODO B3.
3. Runs a pinch analysis (heat_integration.py) on each option and
   converts utilities to electricity under explicit assumptions.

    python ammonia_energy.py            # Ft=8500 kg/h, Fh2=10 kg/h
    python ammonia_energy.py --dtmin 5

All assumptions are constants at the top of the file.
"""

import argparse
import logging
import math
import sys
from pathlib import Path

from pyomo.environ import value

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heat_integration import Stream, pinch, plot_gcc  # noqa: E402

# ------------------------------------------------------------ assumptions
T_CW = 30.0            # C, cooling water supply; heat rejected above
CW_ELEC = 0.02         # kWe per kW rejected to cooling water (pumps, fans)
T_AMB = 25.0           # C, heat-pump source
HP_CARNOT_FRAC = 0.5   # heat pump COP = 0.5 * Carnot
MVR_COP = 8.0          # MVR: kW heat per kWe for ~20 K lift on steam
CP_BRINE = 3.3         # kJ/kg/K, concentrated brine (S13)
CP_WATER = 4.2         # kJ/kg/K
DH_NH3_40C = 1100.0    # kJ/kg, latent heat NH3 at 40 C (15.5 bar)
MEMBRANE_KWH_PER_KG_N = 7.1     # paper / Iddya et al. 2020
N_STAGES_STRIP = 10    # theoretical stripping stages
RECOVERY = 0.99        # NH3 recovered from S13
ALPHA_RECT = 6.0       # NH3/H2O relative volatility in the rectifier
R_FACTOR = 1.5         # R = 1.5 Rmin
X_D = 0.995            # anhydrous NH3 purity (mole fraction)
P_RECT_T_BOT = 198.0   # C, water boiling point at 15 bar (rectifier sump)
T_RECT_COND = 40.0     # C, NH3 condensing at 15.5 bar
T_RECT_FEED = 150.0    # C, rectifier feed after bottoms/feed exchanger
# Brine pre-concentration B6 (TODO A9; docs/references.md R7). The feed is
# already ~103 g/kg TDS, above the ~70-80 g/L conventional-RO limit, so
# water removal is thermal: mechanical vapour compression ~24 kWh/m3
# (70 -> 250 g/L); beyond NaCl saturation an evaporator-crystallizer is
# needed, taken as ~35 kWh/m3.
E_MVC = 24.0           # kWh per m3 water removed, below saturation
E_CRYST = 35.0         # kWh per m3 water removed, crystallizing
NACL_SOL = 0.360       # kg NaCl per kg water at 25 C (26.5 wt%)


def henry_k(t_c, p_bar):
    """K = y/x for dilute NH3 in water. Henry solubility from Sander
    (2015): H = 59 mol/(kg bar) at 25 C, d ln H / d(1/T) = 4200 K.
    Salting-out by the brine is ignored (conservative: real K higher)."""
    h = 59.0 * math.exp(4200.0 * (1.0 / (t_c + 273.15) - 1.0 / 298.15))
    return 1.0 / (h * 0.018015 * p_bar)


def dh_vap_water(t_c):
    """kJ/mol, Watson-type fit (40.7 at 100 C)."""
    return 40.65 * ((647.1 - (t_c + 273.15)) / (647.1 - 373.15)) ** 0.38


def kremser_s(n, recovery):
    """Stripping factor giving `recovery` in `n` theoretical stages."""
    lo, hi = 1.0001, 50.0
    for _ in range(200):
        s = 0.5 * (lo + hi)
        rec = (s ** (n + 1) - s) / (s ** (n + 1) - 1)
        lo, hi = (s, hi) if rec < recovery else (lo, s)
    return 0.5 * (lo + hi)


# -------------------------------------------------------------- flowsheet
def solve_ammonia(ft=8500.0, fh2=10.0, t_flash_c=None, s16_water=None):
    import ammonia_flowsheet as A
    if s16_water is not None:
        # water left in the B6 concentrate [kg/h] (.bkp: 1400)
        A.S16_H2O_KGPH = s16_water
    if t_flash_c is not None:
        # C1/B2 flash temperature (the .bkp uses 10 C). At 40-50 C the
        # NH3 product is unchanged but C1 needs no chiller.
        A.T_FLASH = t_flash_c + 273.15
    logging.disable(logging.WARNING)
    import contextlib
    import io
    # dilute concentrates (> ~4000 kg/h water) do not start cold: converge
    # at 3000 kg/h and walk the B6 water target up (continuation)
    target = A.S16_H2O_KGPH
    start = min(target, 3000.0)
    A.S16_H2O_KGPH = start
    m = A.build_model(fh2_kgph=fh2)
    A.S16_H2O_KGPH = target
    s = A.initialize(m)

    def robust_solve():
        for _ in range(4):
            r = s.solve(m, load_solutions=False)
            if r.solution:
                m.solutions.load_from(r)
            if str(r.solver.termination_condition) == "optimal":
                return
        raise RuntimeError(
            f"ammonia flowsheet: {r.solver.termination_condition}")

    with contextlib.redirect_stdout(io.StringIO()):
        robust_solve()
        n = max(1, int((target - start) / 500.0 + 0.999))
        for i in range(1, n + 1):
            w = start + (target - start) * i / n
            m.fs.b6.s16_h2o_target.set_value(w / 3600.0 / A.MW["H2O"])
            robust_solve()
    return m


def brine_concentration(m):
    """Electricity for B6 water removal and a saturation check."""
    import ammonia_flowsheet as A
    feed = A.BRINE_KGPH
    w_in = feed["H2O"]
    w_left = A.S16_H2O_KGPH
    removed = max(w_in - w_left, 0.0)
    nacl = min(feed["Na+"] * 58.44 / 22.99, feed["Cl-"] * 58.44 / 35.45)
    saturated = nacl / max(w_left, 1e-9) > NACL_SOL
    e = E_CRYST if saturated else E_MVC
    salts = sum(v for k, v in feed.items() if k != "H2O")
    return {"water removed [kg/h]": removed,
            "concentrate salts [wt%]": 100 * salts / (salts + w_left),
            "NaCl / water [kg/kg]": nacl / max(w_left, 1e-9),
            "above NaCl saturation": float(saturated),
            "specific energy [kWh/m3]": e,
            "B6 electricity [kW]": removed / 1000.0 * e}


def base_streams(m):
    """Process duties of the existing ammonia flowsheet."""
    fs = m.fs
    c = lambda T: value(T) - 273.15  # noqa: E731
    st = []
    for k in (1, 2, 3):
        ic = getattr(fs, f"ic{k}")
        st.append(Stream(f"intercooler IC{k}",
                         c(ic.control_volume.properties_in[0].temperature),
                         c(ic.control_volume.properties_out[0].temperature),
                         -value(ic.heat_duty[0]) / 1e3, section="ammonia"))
    h1 = fs.h1.control_volume
    st.append(Stream("H1 reactor preheat", c(h1.properties_in[0].temperature),
                     c(h1.properties_out[0].temperature),
                     abs(value(fs.Qh1)) / 1e3, section="ammonia"))
    t_r = c(h1.properties_out[0].temperature)
    st.append(Stream("B1 reactor (isothermal)", t_r + 0.1, t_r,
                     -value(fs.Qr1) / 1e3, section="ammonia"))
    c1 = fs.c1.control_volume
    st.append(Stream("C1 effluent cooler", c(c1.properties_in[0].temperature),
                     c(c1.properties_out[0].temperature),
                     -value(fs.Qc1) / 1e3, section="ammonia"))
    # if IDAES gives H1 as cooling (sign flip at some points), fix kind
    for s_ in st:
        if s_.duty_kw < 0:
            s_.t_in, s_.t_out, s_.duty_kw = s_.t_out, s_.t_in, -s_.duty_kw
    return st


def recovery_streams(option, nh3_kgph, h2o_kgph, s13_kgph, t_s13):
    """Shortcut NH3-recovery design. Returns (streams, shaft_kw, info)."""
    if option == "M":
        kw = nh3_kgph * 14.007 / 17.031 * MEMBRANE_KWH_PER_KG_N
        return [], kw, {"membrane electricity [kW]": kw}

    if option in ("S1", "S3"):
        p, t_b = 1.013, 100.0
    else:                                   # S2 vacuum
        p, t_b = 0.123, 50.0
    k = henry_k(t_b, p)
    s_fac = kremser_s(N_STAGES_STRIP, RECOVERY)
    l_kmol = h2o_kgph / 18.015
    v_kmol = s_fac * l_kmol / k
    q_reb = v_kmol * dh_vap_water(t_b) * 1e3 / 3600.0          # kW
    n_nh3 = RECOVERY * nh3_kgph / 17.031
    w_ov = max(v_kmol - n_nh3, 0.0)                            # kmol/h
    st = [
        Stream("stripper feed preheat", t_s13, t_b - 2.0,
               s13_kgph * CP_BRINE * (t_b - 2.0 - t_s13) / 3600.0,
               section="recovery"),
        Stream("stripper reboiler", t_b, t_b + 0.1, q_reb, section="recovery"),
        Stream("stripper bottoms cooler", t_b, T_CW + 5.0,
               s13_kgph * CP_BRINE * (t_b - T_CW - 5.0) / 3600.0,
               section="recovery"),
        # overhead (water + ~10 mol% NH3) condensed to aqua ammonia; under
        # vacuum the aqua bubble point is low, so condense to 10 C
        Stream("stripper overhead condenser", t_b - 2.0,
               10.0 if option == "S2" else 40.0, q_reb, section="recovery"),
    ]
    shaft = 0.0
    if option == "S3":
        # bottoms MVR raises the stripping steam: reboiler duty supplied
        # by compressor work instead of external heat
        st[1] = Stream("stripper reboiler (MVR-supplied)", t_b, t_b + 0.1,
                       0.0 + 1e-6, section="recovery")
        shaft += q_reb / MVR_COP

    # rectifier to anhydrous NH3 (15 bar)
    x_f = n_nh3 / (n_nh3 + w_ov)
    r_min = (1 / (ALPHA_RECT - 1)) * (X_D / x_f
                                      - ALPHA_RECT * (1 - X_D) / (1 - x_f))
    r = R_FACTOR * max(r_min, 0.1)
    q_cond = (r + 1) * nh3_kgph * RECOVERY * DH_NH3_40C / 3600.0
    w_kg = w_ov * 18.015
    q_reb_r = q_cond + w_kg * CP_WATER * (P_RECT_T_BOT - T_RECT_FEED) / 3600
    st += [
        Stream("rectifier feed preheat", 40.0, T_RECT_FEED,
               (w_kg + nh3_kgph) * CP_WATER * (T_RECT_FEED - 40.0) / 3600,
               section="recovery"),
        Stream("rectifier reboiler (15 bar)", P_RECT_T_BOT,
               P_RECT_T_BOT + 0.1, q_reb_r, section="recovery"),
        Stream("rectifier condenser (NH3, 40 C)", T_RECT_COND + 0.1,
               T_RECT_COND, q_cond, section="recovery"),
        Stream("rectifier bottoms cooler", P_RECT_T_BOT, T_CW + 5.0,
               w_kg * CP_WATER * (P_RECT_T_BOT - T_CW - 5.0) / 3600,
               section="recovery"),
    ]
    info = {"stripper P [bar]": p, "K_NH3": k, "stripping factor": s_fac,
            "stripping vapour [kg/h]": v_kmol * 18.015,
            "overhead NH3 [mol%]": 100 * x_f,
            "rectifier R (=1.5 Rmin)": r}
    return st, shaft, info


# -------------------------------------------------------------- utilities
def electricity(res, streams, shaft_kw, cop_chiller, integrated):
    """kWe for one option, with or without heat integration."""
    if integrated:
        q_hot = res.q_hot_min
        q_chill = res.cold_below(T_CW + 5.0)
        q_cw = res.q_cold_min - q_chill
        t_hot = max((s.t_out for s in streams if s.kind == "cold"),
                    default=T_AMB)
    else:
        q_hot = res.q_hot_no_hi
        q_chill = sum(s.duty_kw * max(0.0, min(1.0, (T_CW + 5.0 - s.t_out)
                                               / (s.t_in - s.t_out)))
                      for s in streams if s.kind == "hot")
        q_cw = res.q_cold_no_hi - q_chill
        t_hot = max((s.t_out for s in streams if s.kind == "cold"),
                    default=T_AMB)
    th, ta = t_hot + 10 + 273.15, T_AMB + 273.15
    cop_hp = max(1.0, HP_CARNOT_FRAC * th / max(th - ta, 1.0))
    return {
        "hot utility [kW]": q_hot,
        "cooling water [kW]": q_cw,
        "chilled (<35 C) [kW]": q_chill,
        "shaft (compressors, MVR, membrane) [kWe]": shaft_kw,
        "heating, electric resistance [kWe]": q_hot,
        f"heating, heat pump COP {cop_hp:.1f} [kWe]": q_hot / cop_hp,
        "cooling water [kWe]": CW_ELEC * q_cw,
        f"chiller COP {cop_chiller:.2f} [kWe]": q_chill / cop_chiller,
        "TOTAL, resistance heating [kWe]":
            shaft_kw + q_hot + CW_ELEC * q_cw + q_chill / cop_chiller,
        "TOTAL, heat-pump heating [kWe]":
            shaft_kw + q_hot / cop_hp + CW_ELEC * q_cw + q_chill / cop_chiller,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fh2", type=float, default=10.0)
    ap.add_argument("--dtmin", type=float, default=10.0)
    ap.add_argument("--s16water", type=float, default=None,
                    help="water left in the B6 concentrate [kg/h] "
                         "(.bkp 1400; 7621 = no concentration)")
    ap.add_argument("--no-b6-energy", action="store_true",
                    help="exclude B6 brine-concentration energy (as in the "
                         "paper and the .bkp)")
    ap.add_argument("--tflash", type=float, default=None,
                    help="C1/B2 flash temperature [C] (default: .bkp 10 C)")
    a = ap.parse_args()

    m = solve_ammonia(fh2=a.fh2, t_flash_c=a.tflash, s16_water=a.s16water)
    fs = m.fs
    nh3 = value(fs.nh3_product_kgph)
    h2o = value(fs.h2o_product_kgph)
    s13 = value(fs.s13_mass_kgph)
    t13 = value(fs.s13.properties[0].temperature) - 273.15
    w_comp = value(fs.W_comp) / 1e3
    cop_ch = value(fs.refrigeration.cop)
    base = base_streams(m)

    print("\n=== ammonia flowsheet duties (IDAES, Ft=8500 kg/h, "
          f"Fh2={a.fh2} kg/h) ===")
    print(f"NH3 product {nh3:.2f} kg/h in {s13:.0f} kg/h S13 at {t13:.1f} C "
          f"({100 * nh3 / s13:.2f} wt%)")
    for s in base:
        print(f"  {s.name:28s} {s.kind:4s} {s.t_in:7.1f} -> {s.t_out:7.1f} C"
              f" {s.duty_kw:9.1f} kW")
    print(f"  {'H2 compressor work':28s} {'':4s} {'':19s} {w_comp:9.1f} kW")
    print(f"  {'R717 chiller on C1 (Canberk)':28s} COP {cop_ch:.2f} -> "
          f"{value(fs.refrigeration.electric_power) / 1e3:.1f} kWe")
    paper = sum(s.duty_kw for s in base) + w_comp
    print(f"  paper convention (sum |duties|, COP 1): {paper:.1f} kW-eq")

    b6 = brine_concentration(m)
    print("\nB6 brine pre-concentration (absent from .bkp and paper):")
    for k_, v in b6.items():
        print(f"  {k_:34s} {v:10.2f}")
    b6_kw = 0.0 if a.no_b6_energy else b6["B6 electricity [kW]"]

    out_dir = Path(__file__).resolve().parents[1] / "figures"
    rows = {}
    for opt, label in (("M", "membrane (paper)"),
                       ("S1", "steam stripper + rectifier"),
                       ("S2", "vacuum stripper + rectifier"),
                       ("S3", "steam stripper w/ MVR + rectifier")):
        rs, shaft, info = recovery_streams(opt, nh3, h2o, s13, t13)
        streams = base + rs
        res = pinch(streams, a.dtmin)
        sh = w_comp + shaft + b6_kw
        no = electricity(res, streams, sh, cop_ch, integrated=False)
        hi = electricity(res, streams, sh, cop_ch, integrated=True)
        rows[label] = (no, hi, res, info)
        print(f"\n--- option {opt}: {label} ---")
        for k_, v in info.items():
            print(f"  {k_:34s} {v:10.3f}")
        for s in rs:
            print(f"  {s.name:34s} {s.kind:4s} {s.t_in:6.1f}->{s.t_out:6.1f} C"
                  f" {s.duty_kw:8.1f} kW")
        print("  " + res.summary())
        tag = "" if a.tflash is None else f"_flash{a.tflash:.0f}C"
        plot_gcc(res, out_dir / f"ammonia_gcc_{opt}{tag}.png",
                 f"Ammonia side + recovery {opt}: grand composite curve")

    print("\n=== summary: electricity per kg NH3 (kWh/kg) ===")
    hdr = f"{'option':36s} {'no HI, resist.':>15s} {'HI, resist.':>12s}" \
          f" {'HI, heat pump':>14s} {'hot util HI kW':>15s}"
    print(hdr)
    for label, (no, hi, res, _) in rows.items():
        print(f"{label:36s} "
              f"{no['TOTAL, resistance heating [kWe]'] / nh3:15.1f} "
              f"{hi['TOTAL, resistance heating [kWe]'] / nh3:12.1f} "
              f"{hi['TOTAL, heat-pump heating [kWe]'] / nh3:14.1f} "
              f"{res.q_hot_min:15.1f}")
    print(f"\nSWEEP s16water={a.s16water} NH3={nh3:.2f} B6={b6_kw:.1f}kW "
          + " ".join(f"{lab.split()[0]}:"
                     f"{hi['TOTAL, heat-pump heating [kWe]'] / nh3:.2f}"
                     for lab, (no, hi, res, _) in rows.items()))
    print("\n(kWh per kg NH3 recovered; ammonia section incl. H2 compression,"
          " excl. electrolysis. Membrane option: its 7.1 kWh/kg N is pure"
          " electricity and cannot be heat-integrated.)")


if __name__ == "__main__":
    main()
