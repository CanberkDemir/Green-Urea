"""
Direct electrochemical nitrate reduction (NO3RR) screening model.

Alternative to the current ammonia section (water electrolysis -> H2
compression -> Ni2P / Pd-Cu hydrogenation PFR -> flash): reduce the
nitrate in the brine directly at a cathode,

    cathode  NO3- + 6 H2O + 8 e-  -> NH3 + 9 OH-      E0 = +0.69 V vs RHE
    side     2 H2O + 2 e-         -> H2 + 2 OH-        (HER, lowers FE)
    anode    2 H2O                -> O2 + 4 H+ + 4 e-  E0 = +1.23 V vs RHE

so the reversible cell voltage is ~0.54 V, against 1.23 V for making H2
first. The electron count per NH3 (8) is the same in both routes.

Cell concept (assumed, see docs/electro_no3rr_survey.md):
  * divided flow cell, cation-exchange membrane (CEM), brine on the
    cathode side only -> Cl- never reaches the anode, so no Cl2/ClO-
    (which would oxidise the NH3 product back to N2);
  * acidic anolyte (dilute H2SO4, IrO2 anode). H+ crosses the CEM, so
    the net catholyte change is +1 OH- per NH3: the outlet is pH ~13,
    i.e. free NH3, ready for stripping, and no chemical is consumed;
  * stages in series (cascade). Each stage runs at a fixed fraction
    `phi` of its inlet nitrate limiting current, so FE stays high while
    nitrate is depleted. This is the main design lever for a dilute feed.

Per stage the local NH3 partial current is
    j_N = min(FE_max * j, 8 F k_m c)          (mass-transfer cap)
and the rest of the current goes to HER. The stage is integrated along
the electrode area (plug flow). Cell voltage per stage:
    V = E_rev + eta_c + b_a*log10(j/j0_a) + j*ASR

Also: a CO2 + NO3- -> urea co-electrolysis estimate (literature FE/j,
no mechanistic model - the chemistry is not mature enough for one).

    python electro_no3rr.py                 # three parameter cases, raw brine
    python electro_no3rr.py --feed b6       # after B6 concentration
    python electro_no3rr.py --stages 1      # single stage for comparison
"""

import argparse
import math

F = 96485.0  # C/mol
MW_NH3, MW_NO3, MW_UREA = 17.031, 62.005, 60.055

# Brine at the validation point (ammonia_flowsheet.BRINE_KGPH)
BRINE_KGPH = {"H2O": 7621.015, "Cl-": 453.9, "Na+": 330.565,
              "NO3-": 78.625, "SO4-2": 15.895}
B6_WATER_KGPH = 1400.0   # water left after the B6 concentration step
RHO_BRINE = 1060.0       # kg/m3, raw; ~1200 after B6

E_REV = 1.23 - 0.69      # V, OER minus NO3-/NH3 (both vs RHE)

# (optimistic, central, pessimistic). Ranges from docs/electro_no3rr_survey.md
CASES = {
    "optimistic":  dict(fe_max=0.95, eta_c=0.70, b_a=0.040, j0_a=1e-3,
                        asr=0.5e-4, k_m=5e-5, phi=0.8),
    "central":     dict(fe_max=0.90, eta_c=0.90, b_a=0.050, j0_a=1e-4,
                        asr=1.0e-4, k_m=2e-5, phi=0.7),
    "pessimistic": dict(fe_max=0.80, eta_c=1.20, b_a=0.060, j0_a=1e-5,
                        asr=2.0e-4, k_m=1e-5, phi=0.6),
}
# eta_c: cathode overpotential vs the 0.69 V NO3-/NH3 couple [V]
#        (catalysts typically run at -0.0 to -0.5 V vs RHE)
# b_a, j0_a: OER Tafel slope [V/dec] and exchange current [A/m2] (IrO2)
# asr: membrane + catholyte gap + contact [ohm m2] (1e-4 = 1 ohm cm2)
# k_m: mass-transfer coefficient in the cathode channel [m/s]
# phi: stage current / stage-inlet limiting current

STACK_CAPEX_PER_M2 = (3000.0, 6000.0, 12000.0)  # $/m2 electrode, rough


def feed(kind):
    """Return (nitrate mol/s, liquid m3/s, NaCl molarity) for the cell feed."""
    n_no3 = BRINE_KGPH["NO3-"] / MW_NO3 / 3.6           # mol/s
    salts = sum(v for k, v in BRINE_KGPH.items() if k != "H2O")
    if kind == "raw":
        mass, rho = BRINE_KGPH["H2O"] + salts, RHO_BRINE
    else:
        mass, rho = B6_WATER_KGPH + salts, 1200.0
    q = mass / rho / 3600.0                              # m3/s
    c_nacl = BRINE_KGPH["Cl-"] / 35.453 / 3.6 / q / 1e3  # mol/L
    return n_no3, q, c_nacl


def cell_voltage(j, p):
    return (E_REV + p["eta_c"] + p["b_a"] * math.log10(max(j, 1e-9) / p["j0_a"])
            + j * p["asr"])


def design(p, kind="raw", stages=4, x_target=0.95, n_seg=400):
    """Size a cascade of `stages` cells reaching nitrate conversion x_target."""
    n0, q, c_nacl = feed(kind)
    c_in, c_out_final = n0 / q, n0 / q * (1.0 - x_target)   # mol/m3
    # equal log-conversion per stage
    ratio = (c_out_final / c_in) ** (1.0 / stages)
    res, c = [], c_in
    for s in range(stages):
        c_goal = c * ratio
        j = p["phi"] * 8 * F * p["k_m"] * c                # A/m2, constant in stage
        area = q_e = 0.0
        dc = (c - c_goal) / n_seg
        cc = c
        for _ in range(n_seg):                              # plug flow along area
            jn = min(p["fe_max"] * j, 8 * F * p["k_m"] * (cc - dc / 2))
            da = 8 * F * q * dc / jn                        # m2 for this slice
            area += da
            q_e += j * da                                   # A
            cc -= dc
        v = cell_voltage(j, p)
        res.append(dict(stage=s + 1, c_in=c, c_out=c_goal, j=j, area=area,
                        current=q_e, fe=8 * F * q * (c - c_goal) / q_e,
                        v=v, power_kw=q_e * v / 1e3))
        c = c_goal
    nh3_kgph = n0 * x_target * MW_NH3 * 3.6
    tot = dict(
        area=sum(r["area"] for r in res),
        current=sum(r["current"] for r in res),
        power_kw=sum(r["power_kw"] for r in res),
        nh3_kgph=nh3_kgph, c_in=c_in, c_nacl=c_nacl, q_m3h=q * 3600)
    tot["kwh_per_kg_nh3"] = tot["power_kw"] / nh3_kgph
    tot["fe"] = 8 * F * n0 * x_target / tot["current"]
    # HER by-product (can be credited if the cathode gas is collected)
    tot["h2_kgph"] = (tot["current"] - 8 * F * n0 * x_target) / (2 * F) \
        * 2.016e-3 * 3600
    tot["oh_out_M"] = n0 * x_target / q / 1e3     # +1 OH- per NH3 (acid anolyte)
    return res, tot


def urea_coelectrolysis(nh3_equiv_kgph, fe=(0.6, 0.3, 0.1),
                        j=(1000.0, 300.0, 100.0), v=(2.8, 3.3, 3.8)):
    """2 NO3- + CO2 + 18 H+ + 16 e- -> CO(NH2)2 + 7 H2O. Literature-range only."""
    urea_kgph = nh3_equiv_kgph / (2 * MW_NH3) * MW_UREA
    out = []
    for f_, j_, v_ in zip(fe, j, v):
        i = urea_kgph / 3.6 / MW_UREA * 16 * F / f_           # A
        out.append(dict(fe=f_, j=j_, v=v_, area=i / j_, power_kw=i * v_ / 1e3,
                        kwh_per_kg_urea=i * v_ / 1e3 / urea_kgph,
                        urea_kgph=urea_kgph))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", choices=["raw", "b6"], default="raw")
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--conversion", type=float, default=0.95)
    a = ap.parse_args()

    n0, q, c_nacl = feed(a.feed)
    print(f"Feed: {a.feed}, {q * 3600:.2f} m3/h, NO3- {n0 / q:.0f} mol/m3 "
          f"({n0 / q / 1e3:.3f} M), NaCl ~{c_nacl:.2f} M; "
          f"max NH3 {n0 * MW_NH3 * 3.6:.2f} kg/h")
    if c_nacl > 5.0:
        print("  WARNING: NaCl near saturation (6.1 M) - cathode scaling risk")
    print(f"Cascade: {a.stages} stage(s), nitrate conversion {a.conversion:.0%}\n")

    for name, p in CASES.items():
        res, t = design(p, a.feed, a.stages, a.conversion)
        print(f"== {name}")
        for r in res:
            print(f"   stage {r['stage']}: j={r['j'] / 10:6.1f} mA/cm2  "
                  f"FE={r['fe']:.2f}  V={r['v']:.2f}  A={r['area']:7.1f} m2  "
                  f"P={r['power_kw']:6.1f} kW")
        capex = STACK_CAPEX_PER_M2[list(CASES).index(name)] * t["area"]
        print(f"   total: {t['nh3_kgph']:.2f} kg/h NH3, area {t['area']:.0f} m2, "
              f"{t['current'] / 1e3:.0f} kA, FE {t['fe']:.2f}, "
              f"{t['power_kw']:.0f} kW -> {t['kwh_per_kg_nh3']:.1f} kWh/kg NH3")
        print(f"   by-product H2 {t['h2_kgph']:.2f} kg/h; outlet OH- "
              f"{t['oh_out_M']:.3f} M (pH ~{14 + math.log10(t['oh_out_M']):.1f}); "
              f"stack capex ~${capex / 1e6:.1f} M\n")

    nh3 = n0 * a.conversion * MW_NH3 * 3.6
    print("== Direct CO2 + NO3- -> urea co-electrolysis (literature-range, "
          "same nitrate)")
    for lab, r in zip(["optimistic", "central", "pessimistic"],
                      urea_coelectrolysis(nh3)):
        print(f"   {lab:11s}: FE {r['fe']:.0%}, {r['j'] / 10:.0f} mA/cm2, "
              f"{r['v']:.1f} V -> {r['urea_kgph']:.1f} kg/h urea, "
              f"area {r['area']:.0f} m2, {r['power_kw']:.0f} kW, "
              f"{r['kwh_per_kg_urea']:.1f} kWh/kg urea")
    print("   (product is ~0.3 wt% urea in ~1.6 M NaCl; urea is non-volatile, "
          "so separating it from the brine is the real cost - not modelled)")


if __name__ == "__main__":
    main()
