"""A9/B3: define the feed-concentration problem and screen staged water removal.

Problem: the brine enters at 103 g/kg TDS (7,621 kg/h water). Every kg of
water that reaches the NH3 stripper must be heated and stripped, so the
stripper duty scales with water flow. Water can be removed in steps, each
with its own physical limit, all evaluated with the Pitzer model
(brine_electrolyte.py, TODO A5):

  RO   high-pressure reverse osmosis, limited by osmotic pressure
       (Pitzer a_w) against an element pressure rating;
  MVC  mechanical-vapour-compression evaporator, compressor lift set by the
       local boiling-point elevation, limited by NaCl saturation
       (with headroom for the NaOH added later for stripping);
  CRY  evaporative crystallizer: keeps going past saturation by removing
       NaCl as solid (optional).

Where the water is removed matters:
  upstream    on the raw brine, before the Ni2P reactor. Lowers dissolved H2
              solubility (salting-out) in a transfer-limited reactor.
  downstream  on the acidic reactor effluent (pH 2, HCl). Ammonia is held as
              non-volatile NH4+, so the condensate is clean water and almost no
              NH3 is lost; the reactor sees unchanged H2 solubility.
Then NaOH raises pH to release NH3 and the 0.08 bar vacuum stripper recovers it.

This is a screening model: no equipment sizing, no scaling/fouling beyond NaCl,
Na2SO4 and NH4Cl treated only through the Pitzer activities.

    python water_removal.py
    python water_removal.py --nh3 11.96 --t-evap 70
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import brine_electrolyte as BE

# ------------------------------------------------------------ assumptions
T_RO = 298.15           # K, RO feed temperature
RO_P_MAX = 120.0        # bar, specialised high-pressure elements (DuPont)
RO_POLARIZATION = 1.10  # wall / bulk osmotic pressure
RO_MARGIN = 13.0        # bar: 10 net driving + 2 channel + 1 permeate
PUMP_EFF = 0.80
ERD_EFF = 0.80          # hydraulic energy recovery on concentrate
MVC_ETA = 0.75          # isentropic efficiency, steam compressor
MVC_DT_HX = 3.0         # K, condensing-boiling approach in the evaporator
MVC_AUX = 2.0           # kWh/m3 pumps and auxiliaries
CRY_AUX = 6.0           # kWh/m3 crystallizer auxiliaries (slurry, centrifuge)
CP_VAP, K_VAP = 1.90, 1.33
SI_MAX = 0.90           # NaCl saturation index allowed before NaOH addition
PH_REACTOR, PH_STRIP = 2.0, 11.0
T_REACTOR = 333.15
HP_FRAC, T_AMB = 0.5, 298.15  # heat-pump COP = 0.5 Carnot (ammonia_energy.py)
T_STRIP_HEAT = 330.15         # K, heat delivered to the 47 C vacuum stripper


@dataclass
class Stream:
    water: float               # kg/h
    ions: dict                 # mol/h
    nh3_total: float = 0.0     # mol/h, NH4+ + NH3 (carried in ions as NH4+)
    nacl_solid: float = 0.0    # kg/h removed as crystal

    def m(self):
        return BE.molality(self.water, self.ions)

    def tds_wtpct(self):
        salts = sum(v * BE.MW[k] / 1000 for k, v in self.ions.items())
        return 100 * salts / (salts + self.water)

    def copy(self, **kw):
        s = Stream(self.water, dict(self.ions), self.nh3_total, self.nacl_solid)
        for k, v in kw.items():
            setattr(s, k, v)
        return s


@dataclass
class Step:
    name: str
    water_in: float
    water_out: float
    kwe: float
    note: str = ""
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------ properties
def osmotic_bar(s: Stream, t=T_RO):
    return BE.osmotic_pressure_bar(s.m(), t)


def _bisect_water(s: Stream, ok, lo=None):
    """Smallest water flow for which ok(stream) holds (ok monotone)."""
    lo = lo or 0.05 * s.water
    hi = s.water
    if not ok(s):
        return s.water
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        hi, lo = (mid, lo) if ok(s.copy(water=mid)) else (hi, mid)
    return hi


def saturation_water(s: Stream, t_k: float, si_max=SI_MAX, then=None):
    """Least water before NaCl SI reaches si_max, checked after the
    downstream chemistry `then(stream)` (HCl/NaOH add Na+ and Cl-)."""
    def ok(x):
        y = then(x) if then else x
        return BE.nacl_saturation_index(y.m(), t_k) <= si_max
    return _bisect_water(s, ok)


# ------------------------------------------------------------ chemistry
def acidify_and_react(s: Stream, conversion: float, selectivity=0.96):
    """HCl to pH 2 plus reaction protons; NO3- -> NH4+ (and N2)."""
    out = s.copy()
    hcl = max(BE.acid_base_dose(s.water, s.ions, PH_REACTOR, T_REACTOR), 0.0)
    no3 = s.ions["NO3-"] * conversion
    hcl += no3 * (1 + selectivity)
    out.ions["Cl-"] += hcl
    out.ions["NO3-"] -= no3
    out.ions["NH4+"] = out.ions.get("NH4+", 0.0) + selectivity * no3
    out.nh3_total = selectivity * no3
    # protons: those consumed by reaction are gone; the pH-set inventory
    # remains as H+/HSO4- and is captured by charge balance in add_base()
    out.ions["H+"] = hcl - no3 * (1 + selectivity)
    return out, hcl


def add_base(s: Stream, t_k=T_REACTOR, ph=PH_STRIP):
    """NaOH to strip pH. Returns (stream with NH3 freed, NaOH mol/h)."""
    # free H+ is left out: its charge then shows up as an anion excess,
    # which acid_base_dose returns as extra base to neutralise it
    ions = {k: v for k, v in s.ions.items() if k != "H+"}
    ions["NH4+"] = s.nh3_total
    naoh = max(-BE.acid_base_dose(s.water, ions, ph, t_k), 0.0)
    out = s.copy()
    out.ions = dict(ions)
    out.ions["Na+"] += naoh
    mm, _ = BE.speciate(s.water, out.ions, ph, t_k)
    out.ions["NH4+"] = mm["NH4+"] * s.water
    return out, naoh


# ------------------------------------------------------------ unit steps
def ro(s: Stream, p_max=RO_P_MAX):
    def ok(x):
        return osmotic_bar(x) * RO_POLARIZATION + RO_MARGIN <= p_max
    w = _bisect_water(s, ok)
    out = s.copy(water=w)
    salts = s.tds_wtpct() / 100
    q_feed = s.water / (1 - salts) / 1070 / 3600          # m3/s
    q_brine = w / (1 - out.tds_wtpct() / 100) / 1100 / 3600
    if w >= s.water:
        return out, Step("RO", s.water, w, 0.0, "no recovery at pressure limit")
    gross = q_feed * (p_max - 1) * 1e5 / PUMP_EFF / 1e3
    erd = q_brine * (p_max - 2 - 1) * 1e5 * ERD_EFF / 1e3
    return out, Step("RO", s.water, w, gross - erd,
                     f"{p_max:.0f} bar, wall pi {osmotic_bar(out)*RO_POLARIZATION:.0f} bar")


def mvc_work_kwh_per_t(a_w: float, t_b: float):
    """Compressor work per tonne vapour at brine boiling T t_b (K)."""
    p1 = a_w * BE.psat_water_bar(t_b)
    p2 = BE.psat_water_bar(t_b + MVC_DT_HX + 0.0)  # condense above brine T
    # vapour must condense above the brine boiling T, which already includes BPE
    r = p2 / p1
    w = CP_VAP * t_b * (r ** ((K_VAP - 1) / K_VAP) - 1) / MVC_ETA  # kJ/kg
    return w / 3.6 + MVC_AUX


def evaporate(s: Stream, w_target: float, t_b: float, crystallize=False,
              slices=40, name="MVC"):
    """Stepwise evaporation to w_target at brine temperature t_b (K).
    With crystallize=True NaCl is removed as solid whenever SI > 1."""
    out = s.copy()
    kwh = 0.0
    nh3_lost = 0.0
    dw = (s.water - w_target) / slices
    if dw <= 0:
        return out, Step(name, s.water, s.water, 0.0, "nothing to remove")
    for _ in range(slices):
        a_w = BE.pitzer(out.m(), t_b).water_activity
        kwh += dw / 1000 * (mvc_work_kwh_per_t(a_w, t_b) + (CRY_AUX if crystallize else 0))
        # NH3 carried by the vapour (relative volatility via Henry, free fraction)
        if out.nh3_total > 0:
            free = BE.free_nh3_fraction(out.m(), PH_REACTOR, t_b)
            k_nh3 = henry_kpx(t_b) / (a_w * BE.psat_water_bar(t_b) * 1e5)
            x = out.nh3_total / (out.water / 18.015 * 1000)
            loss = k_nh3 * free * x * dw / 18.015 * 1000
            out.nh3_total -= loss
            out.ions["NH4+"] -= loss
            nh3_lost += loss
        out.water -= dw
        if crystallize:
            while BE.nacl_saturation_index(out.m(), t_b) > 1.0:
                d = 0.002 * min(out.ions["Na+"], out.ions["Cl-"])
                out.ions["Na+"] -= d
                out.ions["Cl-"] -= d
                out.nacl_solid += d * BE.MW["NaCl"] / 1000
    bpe = BE.boiling_point_elevation_k(out.m(), a_w * BE.psat_water_bar(t_b))
    return out, Step(name, s.water, out.water, kwh,
                     f"T {t_b-273.15:.0f} C, final BPE {bpe:.1f} K"
                     + (f", NaCl solid {out.nacl_solid:.0f} kg/h" if crystallize else ""),
                     {"nh3_lost_kgph": nh3_lost * BE.MW["NH3"] / 1000})


def henry_kpx(t_k):
    """NH3 Henry Kpx (Pa) consistent with ammonia_energy.henry_k."""
    h = 59.0 * math.exp(4200.0 * (1.0 / t_k - 1.0 / 298.15))
    return 1e5 / (h * 0.018015)


def strip_heat_kwth(nh3_kgph, water_kgph, salts_kgph):
    from ammonia_energy import recovery_streams
    from heat_integration import pinch
    st, _, info = recovery_streams("S2", nh3_kgph, water_kgph,
                                   water_kgph + salts_kgph + nh3_kgph, 40)
    reb = next(x.duty_kw for x in st if x.name == "stripper reboiler")
    return pinch(st, 10).q_hot_min, reb


def heat_to_kwe(q):
    cop = HP_FRAC * T_STRIP_HEAT / (T_STRIP_HEAT - T_AMB)
    return q / cop


# ------------------------------------------------------------ routes
def feed_stream():
    w, n = BE.kgph_to_mol(BE.FEED_KGPH)
    return Stream(w, n)


def route(placement: str, stages: tuple, conversion: float, t_evap: float):
    s = feed_stream()
    steps, hcl = [], 0.0
    if placement == "downstream":
        s, hcl = acidify_and_react(s, conversion)
    t_k = t_evap + 273.15
    for st in stages:
        if st == "RO":
            s, step = ro(s)
        elif st == "MVC":
            later = ((lambda x: add_base(x)[0]) if placement == "downstream" else
                     (lambda x: add_base(acidify_and_react(x, conversion)[0])[0]))
            w_sat = saturation_water(s, t_k, then=later)
            s, step = evaporate(s, max(w_sat, 1.0), t_k)
        elif st == "CRY":
            s, step = evaporate(s, 0.5 * s.water, t_k, crystallize=True, name="CRY")
        steps.append(step)
    h2_ratio = 1.0
    if placement == "upstream":
        h2_ratio = h2_solubility_ratio(s)
        s, hcl = acidify_and_react(s, conversion)
    s_b, naoh = add_base(s)
    si = BE.nacl_saturation_index(s_b.m(), T_REACTOR)
    nh3 = s.nh3_total * BE.MW["NH3"] / 1000
    salts = sum(v * BE.MW[k] / 1000 for k, v in s_b.ions.items())
    q_strip, reb = strip_heat_kwth(max(nh3, 1e-6), s.water, salts)
    return dict(placement=placement, stages="+".join(stages) or "none",
                steps=steps, water=s.water, tds=s.tds_wtpct(), nh3=nh3,
                hcl=hcl * BE.MW["HCl"] / 1000, naoh=naoh * BE.MW["NaOH"] / 1000,
                si_after_base=si, h2_ratio=h2_ratio,
                removal_kwe=sum(x.kwe for x in steps), strip_kwth=q_strip,
                reboiler=reb, removed=feed_stream().water - s.water, nh3_lost=sum(x.extra.get("nh3_lost_kgph", 0) for x in steps))


def h2_solubility_ratio(s: Stream):
    """Equilibrium H2 at 27 bar/60 C relative to the raw feed (salting-out)."""
    from ni2p_dossier import hydrogen_equilibrium
    def c(st):
        m_nacl = st.ions["Cl-"] / st.water
        ps = BE.psat_water_bar(T_REACTOR)
        return hydrogen_equilibrium(T_REACTOR, 2.7, (27 - ps) / 10, ps / 10, m_nacl, 983.0)
    return c(s) / c(feed_stream())


# ------------------------------------------------------------ report
def define_problem():
    s = feed_stream()
    m = s.m()
    print("== Feed-concentration problem (Pitzer, A5) ==")
    print(f"water {s.water:.0f} kg/h, TDS {s.tds_wtpct():.2f} wt%, "
          f"I = {BE.pitzer(m).ionic_strength:.2f} mol/kg")
    for t in (298.15, 333.15):
        a = BE.pitzer(m, t)
        print(f"  {t-273.15:3.0f} C: phi {a.osmotic_coefficient:.3f}, a_w {a.water_activity:.4f}, "
              f"pi {BE.osmotic_pressure_bar(m, t):5.1f} bar, NaCl SI {BE.nacl_saturation_index(m, t):.3f}")
    print(f"  BPE at 1 atm {BE.boiling_point_elevation_k(m, 1.01325):.2f} K")
    for t in (298.15, 343.15):
        print(f"  water at NaCl saturation (SI=1), {t-273.15:.0f} C: "
              f"{saturation_water(s, t, 1.0):.0f} kg/h")
    full = s.ions["NO3-"] * 0.96 * BE.MW["NH3"] / 1000
    print(f"  NH3 at full conversion (96% sel.): {full:.2f} kg/h "
          f"-> {100*full/(s.water+full):.3f} wt% in un-concentrated water")
    ro_s, step = ro(s)
    print(f"  RO at {RO_P_MAX:.0f} bar reaches {ro_s.water:.0f} kg/h water ({step.note})\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--conversion", type=float, default=1.0)
    ap.add_argument("--t-evap", type=float, default=70.0, help="evaporator brine T, C")
    a = ap.parse_args()
    define_problem()
    cases = [("upstream", ()), ("upstream", ("RO",)), ("upstream", ("RO", "MVC")),
             ("downstream", ("RO",)), ("downstream", ("RO", "MVC")),
             ("downstream", ("MVC",)), ("downstream", ("RO", "MVC", "CRY"))]
    base = route("upstream", (), a.conversion, a.t_evap)
    hdr = ("placement", "stages", "water", "TDS%", "NH3", "lostNH3", "HCl", "NaOH",
           "SI+base", "H2sol", "rem_kWe", "strip_kWth", "strip_kWe", "total_kWe")
    print("== Staged water removal (conversion %.2f, evaporator %.0f C) ==" % (a.conversion, a.t_evap))
    print(" ".join(f"{h:>10}" for h in hdr))
    for pl, stg in cases:
        r = route(pl, stg, a.conversion, a.t_evap)
        se = heat_to_kwe(r["strip_kwth"])
        row = (r["placement"], r["stages"], r["water"], r["tds"], r["nh3"], r["nh3_lost"],
               r["hcl"], r["naoh"], r["si_after_base"], r["h2_ratio"], r["removal_kwe"],
               r["strip_kwth"], se, r["removal_kwe"] + se)
        print(" ".join(f"{x:>10.3g}" if isinstance(x, float) else f"{x:>10}" for x in row))
        for st in r["steps"]:
            pass
        if r["removed"] > 1:
            saved = base["strip_kwth"] - r["strip_kwth"]
            print(f"{'':>12}stripper heat saved {saved:.0f} kWth = {1000*saved/r['removed']:.1f} kWh_th/m3 removed "
                  f"(~{1000*heat_to_kwe(saved)/r['removed']:.1f} kWh_e/m3 via heat pump) vs "
                  f"{1000*r['removal_kwe']/r['removed']:.1f} kWh_e/m3 spent; reboiler {r['reboiler']:.0f} kW")
        for st in r["steps"]:
            print(f"{'':>12}{st.name}: {st.water_in:.0f} -> {st.water_out:.0f} kg/h, "
                  f"{st.kwe:.1f} kWe ({st.note})")


if __name__ == "__main__":
    main()
