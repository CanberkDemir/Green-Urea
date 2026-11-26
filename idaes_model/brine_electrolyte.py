"""A5: open-parameter Pitzer electrolyte model for the nitrate brine.

Aspen's ELECNRTL (APV140) parameters are licensed, so this module uses the
published Pitzer ion-interaction model instead. It covers the species the
Ni2P route actually produces: Na+, NH4+, H+ / Cl-, NO3-, SO4-2, HSO4-, OH-.

What it gives (all as plain Python, no IDAES dependency):
  * osmotic coefficient, water activity and osmotic pressure
  * ion activity coefficients
  * boiling temperature / boiling-point elevation of the brine
  * NaCl saturation index (precipitation limit for water removal)
  * NH4+/NH3 speciation (what is strippable, what is not)
  * HCl dose to hold the reactor pH and NaOH dose to release NH3

Parameter sources (25 C binary values): Pitzer & Mayorga (1973), Kim &
Frederick (1988), Harvie, Moller & Weare (1984). NaCl temperature
derivatives: Silvester & Pitzer (1977), applied linearly (adequate to
~100 C). Other binaries are held at their 25 C values; mixing terms use
constant theta (no unsymmetric E-theta) and psi = 0. Validity: I <~ 6 mol/kg.
Not included: excess enthalpy (heats of dilution) - the H1/C1 duty gap in A1
still needs L-phi data, see TODO A5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

R = 8.314462618
MW = {"H2O": 18.015, "Na+": 22.990, "NH4+": 18.038, "H+": 1.008,
      "Cl-": 35.453, "NO3-": 62.005, "SO4-2": 96.063, "HSO4-": 97.071,
      "OH-": 17.007, "NH3": 17.031, "HCl": 36.461, "NaOH": 39.997,
      "NaCl": 58.443}
Z = {"Na+": 1, "NH4+": 1, "H+": 1, "Cl-": -1, "NO3-": -1, "SO4-2": -2,
     "HSO4-": -1, "OH-": -1}

# (beta0, beta1, Cphi) at 25 C, alpha1 = 2
BINARY = {
    ("Na+", "Cl-"): (0.0765, 0.2664, 0.00127),
    ("Na+", "NO3-"): (0.0068, 0.1783, -0.00072),
    ("Na+", "SO4-2"): (0.01958, 1.113, 0.00497),
    ("Na+", "HSO4-"): (0.0454, 0.398, 0.0),
    ("Na+", "OH-"): (0.0864, 0.253, 0.0044),
    ("NH4+", "Cl-"): (0.0522, 0.1918, -0.00301),
    ("NH4+", "NO3-"): (-0.0154, 0.112, -0.00003),
    ("NH4+", "SO4-2"): (0.0409, 0.6585, -0.00116),
    ("H+", "Cl-"): (0.1775, 0.2945, 0.0008),
    ("H+", "NO3-"): (0.1119, 0.3206, 0.0010),
    ("H+", "SO4-2"): (0.0298, 0.0, 0.0438),
    ("H+", "HSO4-"): (0.2065, 0.5556, 0.0),
}
DBINARY_DT = {("Na+", "Cl-"): (7.159e-4, 7.00e-4, -1.05e-4)}  # per K
THETA = {frozenset(("Cl-", "NO3-")): 0.016, frozenset(("Cl-", "SO4-2")): 0.02,
         frozenset(("Cl-", "OH-")): -0.05, frozenset(("H+", "Na+")): 0.036,
         frozenset(("Na+", "NH4+")): -0.012}
B_PITZER = 1.2
ALPHA = 2.0

# NaCl solubility in pure water, mol/kg (CRC), for Ksp(T) calibration
_NACL_SAT = ((0.0, 6.10), (25.0, 6.15), (50.0, 6.28), (75.0, 6.45), (100.0, 6.69))


def a_phi(t_k: float) -> float:
    """Debye-Hueckel osmotic slope; fit to tabulated 0-100 C values."""
    t = t_k - 273.15
    return 0.3770 + 5.08e-4 * t + 3.1e-6 * t * t


def _params(c, a, t_k):
    b0, b1, cp = BINARY.get((c, a), (0.0, 0.0, 0.0))
    d = DBINARY_DT.get((c, a))
    if d:
        dt = t_k - 298.15
        b0, b1, cp = b0 + d[0] * dt, b1 + d[1] * dt, cp + d[2] * dt
    return b0, b1, cp


def _g(x):
    return 2 * (1 - (1 + x) * math.exp(-x)) / (x * x)


def _gp(x):
    return -2 * (1 - (1 + x + x * x / 2) * math.exp(-x)) / (x * x)


@dataclass
class Activity:
    ionic_strength: float
    osmotic_coefficient: float
    water_activity: float
    gamma: dict


def pitzer(m: dict, t_k: float = 298.15) -> Activity:
    """m: molality of each ion (mol/kg water)."""
    m = {k: v for k, v in m.items() if k in Z and v > 0}
    cats = [k for k in m if Z[k] > 0]
    ans = [k for k in m if Z[k] < 0]
    ion_i = 0.5 * sum(v * Z[k] ** 2 for k, v in m.items())
    if ion_i <= 0:
        return Activity(0.0, 1.0, 1.0, {})
    si = math.sqrt(ion_i)
    zsum = sum(v * abs(Z[k]) for k, v in m.items())
    aph = a_phi(t_k)
    x = ALPHA * si
    bphi, bb, bprime, cc = {}, {}, {}, {}
    for c in cats:
        for a in ans:
            b0, b1, cp = _params(c, a, t_k)
            bphi[c, a] = b0 + b1 * math.exp(-x)
            bb[c, a] = b0 + b1 * _g(x)
            bprime[c, a] = b1 * _gp(x) / ion_i
            cc[c, a] = cp / (2 * math.sqrt(abs(Z[c] * Z[a])))
    th = lambda i, j: THETA.get(frozenset((i, j)), 0.0)  # noqa: E731

    s = -aph * ion_i ** 1.5 / (1 + B_PITZER * si)
    s += sum(m[c] * m[a] * (bphi[c, a] + zsum * cc[c, a]) for c in cats for a in ans)
    for grp in (cats, ans):
        for i, p in enumerate(grp):
            for q in grp[i + 1:]:
                s += m[p] * m[q] * th(p, q)
    msum = sum(m.values())
    phi = 1 + 2 * s / msum
    a_w = math.exp(-phi * MW["H2O"] / 1000 * msum)

    f = -aph * (si / (1 + B_PITZER * si) + 2 / B_PITZER * math.log(1 + B_PITZER * si))
    f += sum(m[c] * m[a] * bprime[c, a] for c in cats for a in ans)
    sum_mc = sum(m[c] * m[a] * cc[c, a] for c in cats for a in ans)
    gam = {}
    for i in m:
        z = Z[i]
        ln = z * z * f + abs(z) * sum_mc
        if z > 0:
            ln += sum(m[a] * (2 * bb[i, a] + zsum * cc[i, a]) for a in ans)
            ln += sum(2 * m[c] * th(i, c) for c in cats if c != i)
        else:
            ln += sum(m[c] * (2 * bb[c, i] + zsum * cc[c, i]) for c in cats)
            ln += sum(2 * m[a] * th(i, a) for a in ans if a != i)
        gam[i] = math.exp(ln)
    return Activity(ion_i, phi, a_w, gam)


# ------------------------------------------------------------ properties
def psat_water_bar(t_k: float) -> float:
    """Antoine (NIST, 255-373 K), extended to ~400 K."""
    return 10 ** (4.6543 - 1435.264 / (t_k - 64.848))


def osmotic_pressure_bar(m: dict, t_k: float = 298.15) -> float:
    a_w = pitzer(m, t_k).water_activity
    v_w = MW["H2O"] / 997.0 / 1000  # m3/mol
    return -R * t_k * math.log(a_w) / v_w / 1e5


def boiling_point_k(m: dict, p_bar: float) -> float:
    """T at which a_w * Psat(T) = P."""
    lo, hi = 273.2, 450.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        if pitzer(m, t).water_activity * psat_water_bar(t) > p_bar:
            hi = t
        else:
            lo = t
    return 0.5 * (lo + hi)


def boiling_point_elevation_k(m: dict, p_bar: float) -> float:
    lo, hi = 273.2, 450.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        lo, hi = (lo, t) if psat_water_bar(t) > p_bar else (t, hi)
    return boiling_point_k(m, p_bar) - 0.5 * (lo + hi)


def _interp_sat(t_c):
    pts = _NACL_SAT
    if t_c <= pts[0][0]:
        return pts[0][1]
    for (t0, m0), (t1, m1) in zip(pts, pts[1:]):
        if t_c <= t1:
            return m0 + (m1 - m0) * (t_c - t0) / (t1 - t0)
    return pts[-1][1]


def ln_ksp_nacl(t_k: float) -> float:
    """Self-consistent with this Pitzer set: pure-water saturation."""
    ms = _interp_sat(t_k - 273.15)
    g = pitzer({"Na+": ms, "Cl-": ms}, t_k).gamma
    return math.log(ms * g["Na+"] * ms * g["Cl-"])


def nacl_saturation_index(m: dict, t_k: float) -> float:
    """>1 means NaCl precipitates. Ratio of IAP to Ksp (not log)."""
    if m.get("Na+", 0) <= 0 or m.get("Cl-", 0) <= 0:
        return 0.0
    g = pitzer(m, t_k).gamma
    return math.exp(math.log(m["Na+"] * g["Na+"] * m["Cl-"] * g["Cl-"]) - ln_ksp_nacl(t_k))


# -------------------------------------------------------------- acid/base
def pka_nh4(t_k: float) -> float:
    """Bates & Pinching (1949)."""
    return 0.09018 + 2729.92 / t_k


def pka_hso4(t_k: float) -> float:
    """Marshall & Jones (1966) second dissociation of H2SO4."""
    return -5.3505 + 0.0183412 * t_k + 557.2461 / t_k


def pkw(t_k: float) -> float:
    """Water ion product, simple fit (14.00 at 25 C, 13.02 at 60 C)."""
    return 4470.99 / t_k - 6.0875 + 0.01706 * t_k


def free_nh3_fraction(m: dict, ph: float, t_k: float) -> float:
    """Fraction of total ammonia present as volatile NH3 (gamma_NH3 = 1)."""
    g = pitzer(m, t_k).gamma.get("NH4+", 1.0)
    return 1.0 / (1.0 + 10 ** (pka_nh4(t_k) - ph) / g)


def speciate(water_kg: float, ions_mol: dict, ph: float, t_k: float, iters: int = 30):
    """Split total sulfate (SO4-2 + HSO4-) and total ammonia (NH4+ + NH3) at
    a fixed pH. Returns (molalities incl. H+/OH-, nh3_mol). Charge balance is
    NOT enforced: the acid/base dose that makes it close is computed by
    acid_base_dose(), which is how pH is actually held in practice."""
    so4_t = ions_mol.get("SO4-2", 0.0) + ions_mol.get("HSO4-", 0.0)
    nh_t = ions_mol.get("NH4+", 0.0) + ions_mol.get("NH3", 0.0)
    base = {k: v / water_kg for k, v in ions_mol.items()
            if k not in ("SO4-2", "HSO4-", "NH4+", "NH3", "H+", "OH-")}
    g = {}
    m = dict(base)
    for _ in range(iters):
        a_h = 10 ** -ph
        gh, goh = g.get("H+", 1.0), g.get("OH-", 1.0)
        gs, ghs, gn = g.get("SO4-2", 1.0), g.get("HSO4-", 1.0), g.get("NH4+", 1.0)
        # HSO4- = H+ + SO4-2 ;  K = aH aSO4 / aHSO4
        r_s = 10 ** -pka_hso4(t_k) * ghs / (a_h * gs)       # SO4/HSO4
        r_n = 10 ** -pka_nh4(t_k) * gn / a_h                # NH3/NH4
        m = dict(base)
        m["H+"] = a_h / gh
        m["OH-"] = 10 ** -pkw(t_k) / a_h / goh
        m["SO4-2"] = so4_t / water_kg * r_s / (1 + r_s)
        m["HSO4-"] = so4_t / water_kg / (1 + r_s)
        m["NH4+"] = nh_t / water_kg / (1 + r_n)
        g = pitzer(m, t_k).gamma
    nh3 = nh_t - m["NH4+"] * water_kg
    return m, nh3


def acid_base_dose(water_kg: float, ions_mol: dict, ph: float, t_k: float) -> float:
    """Net strong-acid (positive, as HCl) or strong-base (negative, as NaOH)
    mol needed so that the solution sits at `ph`. ions_mol must be the
    *charge-balanced* composition before the dose (H+/OH- may be omitted).
    The dose ion is Cl- for acid, Na+ for base."""
    dose = 0.0
    for _ in range(40):  # fixed point: activities depend weakly on the dose
        ions = dict(ions_mol)
        key = "Cl-" if dose >= 0 else "Na+"
        ions[key] = ions.get(key, 0.0) + abs(dose)
        mm, _ = speciate(water_kg, ions, ph, t_k)
        charge = sum(Z[k] * v for k, v in mm.items()) * water_kg
        dose += charge  # +charge left over -> more acid (Cl-) or less base (Na+)
    return dose


# -------------------------------------------------------------- the feed
FEED_KGPH = {"H2O": 7621.015, "Cl-": 453.9, "Na+": 330.565,
             "NO3-": 78.625, "SO4-2": 15.895}


def kgph_to_mol(kgph: dict, balance: str | None = "Na+") -> tuple[float, dict]:
    """kg/h -> (kg/h water, mol/h ions). The listed feed analysis is 23 mol/h
    (0.16%) anion-rich; `balance` closes electroneutrality on that ion so that
    acid/base doses are not polluted by analytical error."""
    water = kgph["H2O"]
    mol = {k: v / MW[k] * 1000 for k, v in kgph.items() if k != "H2O"}
    if balance:
        q = sum(Z.get(k, 0) * v for k, v in mol.items())
        mol[balance] = mol.get(balance, 0.0) - q / Z[balance]
    return water, mol


def molality(water_kg: float, mol: dict) -> dict:
    return {k: v / water_kg for k, v in mol.items()}


if __name__ == "__main__":
    w, n = kgph_to_mol(FEED_KGPH)
    m = molality(w, n)
    for t in (298.15, 333.15):
        a = pitzer(m, t)
        print(f"T={t-273.15:.0f} C  I={a.ionic_strength:.3f}  phi={a.osmotic_coefficient:.4f}  "
              f"a_w={a.water_activity:.4f}  pi={osmotic_pressure_bar(m, t):.1f} bar  "
              f"SI_NaCl={nacl_saturation_index(m, t):.3f}")
    print(f"BPE at 1.013 bar: {boiling_point_elevation_k(m, 1.01325):.2f} K")
