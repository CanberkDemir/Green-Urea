"""Pintar et al. (1996) Pd-Cu nitrate-disappearance kinetics.

Applied Catalysis B 11, 81-98, DOI 10.1016/S0926-3373(96)00036-7,
Eq. (3) and Table 2. This is an *overall nitrate disappearance* law, not
a nitrate-to-ammonia pathway model. It is deliberately separate from the
Aspen reaction package in ``ammonia_reactions.py``.

Inputs: nitrate concentration in mg/L, hydrogen partial pressure in bar,
and catalyst loading in g/L. Output: nitrate disappearance in mg/(L min).
The fit covers dilute water, 280.5-293 K, 0.09-0.46 bar H2, and
50-200 mg/L initial nitrate. Do not extrapolate it to the 333 K,
concentrated-brine IDAES reactor without additional data.
"""

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class PintarParameters:
    temperature_k: float
    k_app: float  # mg/(g_cat min bar**0.5)
    k_no3: float  # L/mg


# Experimental values from Table 2 (the table labels K_NO3 * 10**2).
TABLE_2 = {
    280.5: PintarParameters(280.5, 13.9, 2.940e-2),
    283.0: PintarParameters(283.0, 16.9, 2.690e-2),
    288.0: PintarParameters(288.0, 23.9, 2.304e-2),
    293.0: PintarParameters(293.0, 33.1, 1.960e-2),
}


def nitrate_disappearance_rate(
    nitrate_mg_l: float,
    h2_partial_pressure_bar: float,
    catalyst_g_l: float,
    parameters: PintarParameters = TABLE_2[293.0],
) -> float:
    """Return -dC_NO3/dt, in mg/(L min), using the published algebraic law.

    The original data do not identify an NH3/N2 branching rate or a
    hydrogen adsorption constant. In the measured pressure range the
    authors therefore use a square-root partial-pressure dependence.
    """
    if min(nitrate_mg_l, h2_partial_pressure_bar, catalyst_g_l) < 0:
        raise ValueError("concentrations, pressure, and catalyst loading must be nonnegative")
    coverage = parameters.k_no3 * nitrate_mg_l
    return (
        catalyst_g_l
        * parameters.k_app
        * coverage / (1.0 + coverage)
        * sqrt(h2_partial_pressure_bar)
    )
