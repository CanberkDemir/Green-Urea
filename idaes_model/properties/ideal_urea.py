"""
Working property package for the ureaF (Bosch-Meiser) flowsheet.

``srk_urea.py`` is the faithful replica of the .bkp's SR-POLAR-degenerate-RKS
basis, but IDAES's cubic EoS needs the compiled ``cubic_roots.so`` external
functions, and on this machine (Fedora aarch64, no IDAES binaries; see the
ammonia README) a self-built library segfaults inside conda-forge IPOPT's
ASL. Until that is resolved the flowsheet runs on this all-native-Pyomo
package instead - same ``get_prop(phases)`` interface, one-line swap in
``urea_flowsheet.py``.

Formulation (the same Ideal + Henry pattern as the ammonia separation-train
package, extended with temperature-dependent Henry constants):

* Vapor: ideal gas, RPP4 cp with ideal-gas formation enthalpies
  ([BKP] DHFORM for UREA/CARB, NIST for the rest).
* Liquid, Raoult species (H2O, UREA, CARB): extended-Antoine saturation
  pressure. UREA/CARB Antoine constants are refit from the .bkp PLXANT
  records (max error 0.0002 in log10 P over 300-550 K).
* Liquid, Henry species (NH3, CO2): mole-fraction Henry's law with a
  van't Hoff temperature dependence H(T) = H_ref exp(-C (1/T - 1/T_ref)),
  so the reactor-pressure flashes at 200 degC and the 40 degC let-down
  flashes are both in a physically sensible regime (a constant Henry
  coefficient cannot span that window).

Enthalpy bookkeeping is deliberately closed:

* Liquid formation enthalpies of UREA/CARB equal DHFORM(ig) minus the
  vaporisation enthalpy implied by their (shared-slope) Antoine fit
  (dHvap = ln(10) R B = 88.0 kJ/mol), so the latent heat of the VLE and
  the ideal-gas basis are mutually consistent. The result for urea,
  -333.6 kJ/mol, lands on the literature solid-urea value (-333.1).
* Henry species use dH_solution = -R C with C from the same literature
  enthalpy of solution as the liquid formation enthalpy (NBS aqueous
  values), so absorption/desorption heat matches the Henry T-dependence.

Resulting reaction heats (liquid basis, 298 K): carbamate formation from
gaseous NH3/CO2 -146 kJ/mol (lit. -117 to -160 for gas -> condensed),
carbamate -> urea + water +12 kJ/mol endothermic (lit. +15 to +23) - the
correct signs and magnitudes for the Bosch-Meiser heat balance.

Deviation from Aspen to document at validation: ideal-gas vapor at the
141 kgf/cm2 synthesis loop (Aspen: RKS fugacities with kij) and
composition-independent Henry/Raoult liquid (Aspen: kij-mixed RKS liquid).
"""

from copy import deepcopy

from pyomo.environ import exp, units as pyunits, Var

from idaes.core import LiquidPhase, VaporPhase, Component, PhaseType as PT
from idaes.models.properties.modular_properties.base.generic_property import (
    StateIndex,
)
from idaes.models.properties.modular_properties.state_definitions import FTPx
from idaes.models.properties.modular_properties.eos.ideal import Ideal
from idaes.models.properties.modular_properties.phase_equil import SmoothVLE
from idaes.models.properties.modular_properties.phase_equil.bubble_dew import (
    IdealBubbleDew,
)
from idaes.models.properties.modular_properties.phase_equil.forms import fugacity
from idaes.models.properties.modular_properties.phase_equil.henry import (
    HenryType,
    henry_units,
)
from idaes.models.properties.modular_properties.pure.ConstantProperties import (
    Constant,
)
from idaes.models.properties.modular_properties.pure import NIST, RPP4


COMPONENTS = ["NH3", "CO2", "H2O", "UREA", "CARB"]

_J = pyunits.J / pyunits.mol
_JK = pyunits.J / pyunits.mol / pyunits.K
T_HENRY_REF = 298.15  # K


class VantHoffH:
    """Henry coefficient with van't Hoff temperature dependence.

    H(T) = henry_ref * exp(-henry_vant_hoff_C * (1/T - 1/298.15 K))

    Same interface as idaes ...phase_equil.henry.ConstantH; parameter_data
    keys: henry_ref {phase: value}, henry_vant_hoff_C {phase: value [K]}.
    For Kpx (volatility) types, C = -dH_solution / R (positive C means the
    species gets less soluble as temperature rises).
    """

    @staticmethod
    def build_parameters(cobj, p, h_type):
        b = cobj.parent_block()
        units = b.get_metadata().derived_units
        h_units = henry_units(h_type, units)
        cobj.add_component(
            "henry_ref_" + p,
            Var(
                initialize=cobj.config.parameter_data["henry_ref"][p],
                doc="Henry coefficient at 298.15 K for phase " + p,
                units=h_units,
            ),
        )
        cobj.add_component(
            "henry_vant_hoff_C_" + p,
            Var(
                initialize=cobj.config.parameter_data["henry_vant_hoff_C"][p],
                doc="van't Hoff coefficient (-dH_sol/R) for phase " + p,
                units=pyunits.K,
            ),
        )

    @staticmethod
    def return_expression(b, p, j, T=None):
        cobj = b.params.get_component(j)
        if T is None:
            T = b.temperature
        H = getattr(cobj, "henry_ref_" + p)
        C = getattr(cobj, "henry_vant_hoff_C_" + p)
        Tref = T_HENRY_REF * pyunits.K
        return H * exp(-C * (1 / T - 1 / Tref))

    @staticmethod
    def dT_expression(b, p, j, T=None):
        cobj = b.params.get_component(j)
        if T is None:
            T = b.temperature
        C = getattr(cobj, "henry_vant_hoff_C_" + p)
        return VantHoffH.return_expression(b, p, j, T) * C / T**2


def _vap_data(cp_abcd, hf, sf, tc, pc):
    """Ideal-gas side: RPP4 cp/enth/entr (same data as srk_urea).

    Critical constants ride along because the VLE formulation needs them
    for equilibrium-temperature bounds; UREA/CARB values are the [BKP]
    TC/PC records.
    """
    a, b, c, d = cp_abcd
    return {
        "enth_mol_ig_comp": RPP4,
        "entr_mol_ig_comp": RPP4,
        "parameter_data": {
            "cp_mol_ig_comp_coeff": {
                "A": (a, _JK),
                "B": (b, pyunits.J / pyunits.mol / pyunits.K**2),
                "C": (c, pyunits.J / pyunits.mol / pyunits.K**3),
                "D": (d, pyunits.J / pyunits.mol / pyunits.K**4),
            },
            "enth_mol_form_vap_comp_ref": (hf, _J),
            "entr_mol_form_vap_comp_ref": (sf, _JK),
            "temperature_crit": (tc, pyunits.K),
            "pressure_crit": (pc, pyunits.Pa),
        },
    }


# Raoult species: extended-Antoine Psat (NIST log10-bar form) + liquid
# cp/density/formation data.
_H2O = {
    "type": Component,
    "pressure_sat_comp": NIST,
    "enth_mol_liq_comp": Constant,
    "dens_mol_liq_comp": Constant,
    "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
    "parameter_data": {
        "mw": (18.0153e-3, pyunits.kg / pyunits.mol),  # [NIST]
        "pressure_sat_comp_coeff": {  # [NIST] Antoine, 255.9-373 K
            "A": (4.6543, None),
            "B": (1435.264, pyunits.K),
            "C": (-64.848, pyunits.K),
        },
        "cp_mol_liq_comp_coeff": (75.4, _JK),  # [NIST] liquid water
        "enth_mol_form_liq_comp_ref": (-285.83e3, _J),  # [NIST]
        "dens_mol_liq_comp_coeff": (55.3e3, pyunits.mol / pyunits.m**3),
    },
}

# UREA and CARB are declared liquid-only. Their [BKP]-implied volatility is
# tiny everywhere the plant operates (PLXANT gives Psat < 800 Pa even at the
# 200 degC reactor), and keeping them out of the VLE set makes the
# bubble/dew subproblems well-posed (a dew-point over a mixture containing
# ~1e-10 bar species is numerically hopeless - same trick as the vapor-only
# H2/N2 in the ammonia package). Consequence to document: Aspen's DEF-OUT
# urea content comes from the DISTL mole-balance spillover, not urea
# volatility, so the B28 surrogate must reproduce that spillover by split
# logic (see urea_flowsheet.py), and R2/CARBD in the reaction package runs
# on liquid carbamate concentration.
_UREA = {
    "type": Component,
    "valid_phase_types": PT.liquidPhase,
    "enth_mol_liq_comp": Constant,
    "dens_mol_liq_comp": Constant,
    "parameter_data": {
        "mw": (60.0553e-3, pyunits.kg / pyunits.mol),
        "cp_mol_liq_comp_coeff": (135.0, _JK),  # urea melt, DIPPR ~407 K
        # DHFORM(ig) - dHvap(PLXANT slope, 88.0 kJ/mol) = -333.6e3; the
        # literature solid-urea value is -333.1e3
        "enth_mol_form_liq_comp_ref": (-333.6e3, _J),
        "dens_mol_liq_comp_coeff": (22.0e3, pyunits.mol / pyunits.m**3),
    },
}

_CARB = {
    "type": Component,
    "valid_phase_types": PT.liquidPhase,
    "enth_mol_liq_comp": Constant,
    "dens_mol_liq_comp": Constant,
    "parameter_data": {
        "mw": (78.0707e-3, pyunits.kg / pyunits.mol),
        "cp_mol_liq_comp_coeff": (160.0, _JK),  # nominal condensed carbamate
        # DHFORM(ig) - 88.0e3 (lit. solid ammonium carbamate ~ -645e3)
        "enth_mol_form_liq_comp_ref": (-631.5e3, _J),
        "dens_mol_liq_comp_coeff": (20.5e3, pyunits.mol / pyunits.m**3),
    },
}

# NH3 is a Raoult species: it enters the plant PURE (NH3-IN, the S41
# recycle) and appears concentrated in the synthesis liquor, so dilute-
# aqueous Henry's law misprices exactly the states that matter - a pure
# liquid NH3 stream under Henry has ~10x too little vapor pressure and
# every mix of it sits on a spurious phase boundary that stalls the
# VLE solves. Antoine Raoult reproduces pure-NH3 VLE correctly; the price
# (documented deviation) is overestimated NH3 volatility from DILUTE
# aqueous liquors, where the real system is Henry-like.
_NH3 = {
    "type": Component,
    "pressure_sat_comp": NIST,
    "enth_mol_liq_comp": Constant,
    "dens_mol_liq_comp": Constant,
    "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
    "parameter_data": {
        "mw": (17.0305e-3, pyunits.kg / pyunits.mol),
        "pressure_sat_comp_coeff": {  # [NIST] Antoine, 239-371 K
            "A": (4.86886, None),
            "B": (1113.928, pyunits.K),
            "C": (-10.409, pyunits.K),
        },
        "cp_mol_liq_comp_coeff": (80.0, _JK),  # liquid NH3
        # dHf_ig - dHvap(Antoine slope at 300 K, 22.9 kJ/mol) = -68.8e3;
        # the literature NH3(l) value is -69.1e3
        "enth_mol_form_liq_comp_ref": (-68.8e3, _J),
        "dens_mol_liq_comp_coeff": (35.0e3, pyunits.mol / pyunits.m**3),
    },
}

_CO2 = {
    "type": Component,
    "henry_component": {
        "Liq": {
            "method": VantHoffH,
            "type": HenryType.Kpx,
            "basis": StateIndex.true,
        },
    },
    "enth_mol_liq_comp": Constant,
    "dens_mol_liq_comp": Constant,
    "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
    "parameter_data": {
        "mw": (44.0095e-3, pyunits.kg / pyunits.mol),
        # [Sander] H_cp(298) = 3.3e-4 mol/m3/Pa -> Kpx = 1.65e8 Pa;
        # dH_sol = -19.9 kJ/mol -> C = 2394 K. Kpx(473 K) = 3.3e9 Pa.
        "henry_ref": {"Liq": 1.65e8},
        "henry_vant_hoff_C": {"Liq": 2394.0},
        "cp_mol_liq_comp_coeff": (75.0, _JK),
        # = dHf_ig - dH_sol [NBS CO2(aq) = -413.8e3]
        "enth_mol_form_liq_comp_ref": (-413.4e3, _J),
        "dens_mol_liq_comp_coeff": (20.0e3, pyunits.mol / pyunits.m**3),
    },
}

# ideal-gas side data (identical numbers to srk_urea.py; [BKP] CPIGDP fits
# for UREA/CARB, RPP/NIST for the rest)
_VAP = {
    "NH3": _vap_data(
        (2.731e1, 2.383e-2, 1.707e-5, -1.185e-8), -45.898e3, -99.05,
        405.5, 113.5e5,
    ),
    "CO2": _vap_data(
        (1.980e1, 7.344e-2, -5.602e-5, 1.715e-8), -393.51e3, 2.90,
        304.2, 73.8e5,
    ),
    "H2O": _vap_data(
        (3.224e1, 1.924e-3, 1.055e-5, -3.596e-9), -241.826e3, -44.42,
        647.3, 221.2e5,
    ),
    "UREA": _vap_data(
        (-2.21554, 0.289238, -2.133e-4, 6.06076e-8), -245.60e3, -286.14,
        1000.0, 1.284e7,  # [BKP]
    ),
    "CARB": _vap_data(
        (-130.051, 0.722429, -2.65925e-4, 1.64947e-8), -543.50e3, -479.02,
        1000.0, 9.32e6,  # [BKP]
    ),
}

_LIQ = {"NH3": _NH3, "CO2": _CO2, "H2O": _H2O, "UREA": _UREA, "CARB": _CARB}


def get_prop(phases=("Vap", "Liq")):
    """Return a GenericParameterBlock configuration dict.

    Args:
        phases: iterable drawn from {"Liq", "Vap"}; the flowsheet uses the
            two-phase variant everywhere (every Aspen block runs VLE on one
            global basis). Single-phase variants exist for testing.
    """
    phases = tuple(phases)
    if not phases or any(p not in ("Liq", "Vap") for p in phases):
        raise ValueError(f"phases must be drawn from ('Liq', 'Vap'), got {phases}")
    vle = "Liq" in phases and "Vap" in phases

    components = {}
    for name in COMPONENTS:
        cfg = deepcopy(_LIQ[name])
        vap = deepcopy(_VAP[name])
        if cfg.get("valid_phase_types") is PT.liquidPhase:
            # liquid-only species: no ideal-gas methods, but the critical
            # constants still ride along (VLE Teq bounds use them)
            for key in ("temperature_crit", "pressure_crit"):
                cfg["parameter_data"][key] = vap["parameter_data"][key]
        else:
            cfg["enth_mol_ig_comp"] = vap["enth_mol_ig_comp"]
            cfg["entr_mol_ig_comp"] = vap["entr_mol_ig_comp"]
            cfg["parameter_data"].update(vap["parameter_data"])
        if not vle:
            cfg.pop("phase_equilibrium_form", None)
            if "Liq" not in phases:
                cfg.pop("henry_component", None)
                cfg.pop("enth_mol_liq_comp", None)
                cfg.pop("dens_mol_liq_comp", None)
                for key in (
                    "henry_ref",
                    "henry_vant_hoff_C",
                    "cp_mol_liq_comp_coeff",
                    "enth_mol_form_liq_comp_ref",
                    "dens_mol_liq_comp_coeff",
                ):
                    cfg["parameter_data"].pop(key, None)
        components[name] = cfg

    phase_cfg = {}
    if "Liq" in phases:
        phase_cfg["Liq"] = {"type": LiquidPhase, "equation_of_state": Ideal}
    if "Vap" in phases:
        phase_cfg["Vap"] = {"type": VaporPhase, "equation_of_state": Ideal}

    config = {
        "components": components,
        "phases": phase_cfg,
        "base_units": {
            "time": pyunits.s,
            "length": pyunits.m,
            "mass": pyunits.kg,
            "amount": pyunits.mol,
            "temperature": pyunits.K,
        },
        "state_definition": FTPx,
        # grid feeds are 5-50 kg/h (~0.03-0.3 mol/s), recycles a few times
        # larger; T spans the 0 degC let-down flashes to the 200 degC loop;
        # P spans 1 bar to the 200 bar pumps. The 150 K floor is NOT a
        # process temperature - the bubble-point variables of CO2-bearing
        # liquids (Henry-dominated) legitimately sit at 180-230 K at low
        # pressure and the initializer fails if the bound clips them.
        "state_bounds": {
            "flow_mol": (0, 1, 500, pyunits.mol / pyunits.s),
            "temperature": (150, 350, 700, pyunits.K),
            "pressure": (1e4, 1e5, 3e7, pyunits.Pa),
        },
        "pressure_ref": (101325, pyunits.Pa),
        "temperature_ref": (298.15, pyunits.K),
    }

    if vle:
        config["phases_in_equilibrium"] = [("Vap", "Liq")]
        config["phase_equilibrium_state"] = {("Vap", "Liq"): SmoothVLE}
        config["bubble_dew_method"] = IdealBubbleDew

    return config
