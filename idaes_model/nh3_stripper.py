"""
Rigorous IDAES check of the vacuum NH3 stripper (TODO B3).

Takes the S13 hydrous-ammonia product of the solved ammonia flowsheet
and strips it in an N-stage equilibrium column (MESH per stage: Mixer +
Flash, same construction as the urea B25 tray column) at vacuum, with the
reboiler duty on the bottom stage solved so that the bottoms keep only
(1 - RECOVERY) of the feed NH3. Compares the result with the Kremser
shortcut used in ammonia_energy.py (option S2).

Thermodynamics: the ammonia two-phase package (ideal liquid + van't Hoff
Henry NH3, ions liquid-only, H2/N2 vapour-only). Salting-out by the brine
is NOT included (ideal liquid) - conservative for stripping.

    python nh3_stripper.py                 # 0.123 bar, 10 stages
    python nh3_stripper.py --p 0.2 --n 8 --s16water 5000
"""

import argparse
import contextlib
import io
import logging
import sys
from pathlib import Path

from pyomo.environ import Constraint, TransformationFactory, value
from pyomo.network import Arc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idaes.core import FlowsheetBlock  # noqa: E402
from idaes.core.solvers import get_solver  # noqa: E402
from idaes.models.unit_models import Flash, Heater, Mixer  # noqa: E402
from idaes.models.unit_models.mixer import MomentumMixingType  # noqa: E402

_OPT = {"tol": 1e-6, "max_iter": 3000, "mu_strategy": "adaptive"}


def water_tsat(p_bar):
    """Antoine (NIST, 255-373 K) inverse: water boiling point [K]."""
    import math
    a, b, c = 4.6543, 1435.264, -64.848
    return b / (a - math.log10(p_bar)) - c


def build_column(m, props, feed_state, p_pa, n_stages, t_feed):
    fs = m.fs
    col = fs.strip = FlowsheetBlock(dynamic=False)
    col.pre = Heater(property_package=props, has_pressure_change=True)
    col.pre.control_volume.properties_out[0].temperature.fix(t_feed)
    col.pre.control_volume.properties_out[0].pressure.fix(p_pa)
    for k in range(1, n_stages + 1):
        mix = Mixer(property_package=props,
                    inlet_list=["liq", "vap"] if 1 < k < n_stages else
                    (["feed", "vap"] if k == 1 else ["liq"]),
                    momentum_mixing_type=MomentumMixingType.none)
        setattr(col, f"mix{k}", mix)
        mix.mixed_state[0].pressure.fix(p_pa)
        fl = Flash(property_package=props, has_pressure_change=True)
        setattr(col, f"tray{k}", fl)
        fl.deltaP.fix(0.0)
        fl.heat_duty.fix(0.0)
        setattr(col, f"a_mt{k}", Arc(source=mix.outlet, destination=fl.inlet))
    col.a_pre = Arc(source=col.pre.outlet, destination=col.mix1.feed)
    for k in range(1, n_stages + 1):
        fl = getattr(col, f"tray{k}")
        if k < n_stages:
            setattr(col, f"a_l{k}", Arc(source=fl.liq_outlet,
                                        destination=getattr(col, f"mix{k + 1}").liq))
        if k > 1:
            setattr(col, f"a_v{k}", Arc(source=fl.vap_outlet,
                                        destination=getattr(col, f"mix{k - 1}").vap))
    TransformationFactory("network.expand_arcs").apply_to(col)
    # feed: copy the S13 state onto the preheater inlet and fix it
    st = col.pre.control_volume.properties_in[0]
    st.flow_mol.fix(value(feed_state.flow_mol))
    for j in st.mole_frac_comp:
        st.mole_frac_comp[j].fix(max(value(feed_state.mole_frac_comp[j]),
                                     1e-12))
    st.temperature.fix(value(feed_state.temperature))
    st.pressure.fix(value(feed_state.pressure))
    return col


def initialize_column(col, n, q_guess_w, solver):
    from idaes.core.util.initialization import propagate_state
    col.pre.initialize(outlvl=50, optarg=_OPT)
    propagate_state(col.a_pre)
    reb = getattr(col, f"tray{n}")
    reb.heat_duty.fix(q_guess_w)
    # seed upflowing vapour: water at the boiling point
    feed = col.pre.outlet
    t_b = value(feed.temperature[0]) + 2
    v_guess = q_guess_w / 42e3
    for k in range(1, n):
        port = getattr(col, f"mix{k}").vap
        port.flow_mol[0].set_value(v_guess)
        for j in port.mole_frac_comp:
            port.mole_frac_comp[j].set_value(1e-6)
        port.mole_frac_comp[0, "H2O"].set_value(1 - 1e-6 * 9)
        port.temperature[0].set_value(t_b)
        port.pressure[0].set_value(value(feed.pressure[0]))
    for sweep in range(3):
        for k in range(1, n + 1):
            mix = getattr(col, f"mix{k}")
            fl = getattr(col, f"tray{k}")
            if k > 1:
                propagate_state(getattr(col, f"a_l{k - 1}"))
            try:
                mix.initialize(outlvl=50, optarg=_OPT)
            except Exception:
                pass
            propagate_state(getattr(col, f"a_mt{k}"))
            try:
                fl.initialize(outlvl=50, optarg=_OPT)
            except Exception:
                pass
        for k in range(2, n + 1):
            propagate_state(getattr(col, f"a_v{k}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.123, help="column P [bar]")
    ap.add_argument("--n", type=int, default=10, help="equilibrium stages")
    ap.add_argument("--recovery", type=float, default=0.99)
    ap.add_argument("--s16water", type=float, default=None)
    ap.add_argument("--tflash", type=float, default=None)
    a = ap.parse_args()
    logging.disable(logging.WARNING)

    import ammonia_energy as AE
    m = AE.solve_ammonia(s16_water=a.s16water, t_flash_c=a.tflash)
    fs = m.fs
    s13 = fs.s13.properties[0]
    nh3_in = value(s13.flow_mol * s13.mole_frac_comp["NH3"])
    p_pa = a.p * 1e5
    t_b = water_tsat(a.p)
    col = build_column(m, fs.props_vle, s13, p_pa, a.n, t_b - 2.0)
    solver = get_solver("ipopt", options=_OPT)

    # shortcut estimate for the initial reboiler duty
    rs, _, info = AE.recovery_streams(
        "S2", value(fs.nh3_product_kgph), value(fs.h2o_product_kgph),
        value(fs.s13_mass_kgph), value(s13.temperature) - 273.15)
    q_short = [s for s in rs if s.name == "stripper reboiler"][0].duty_kw
    with contextlib.redirect_stdout(io.StringIO()):
        initialize_column(col, a.n, q_short * 1e3, solver)
    # solve with fixed duty, then free the duty and impose the recovery
    from pyomo.environ import Block
    for blk in col.component_data_objects(Block, descend_into=True):
        if hasattr(blk, "eps_1_Vap_Liq"):
            blk.eps_1_Vap_Liq = 1.0
    reb = getattr(col, f"tray{a.n}")
    r = solver.solve(col)
    print(f"fixed-duty solve ({q_short:.1f} kW): "
          f"{r.solver.termination_condition}")
    bot = reb.liq_outlet
    col.recovery_spec = Constraint(
        expr=bot.flow_mol[0] * bot.mole_frac_comp[0, "NH3"]
        == (1 - a.recovery) * nh3_in)
    reb.heat_duty.unfix()
    for _ in range(3):
        r = solver.solve(col)
        if str(r.solver.termination_condition) == "optimal":
            break
    print(f"recovery-spec solve: {r.solver.termination_condition}")

    top = col.tray1.vap_outlet
    q_reb = value(reb.heat_duty[0]) / 1e3
    q_pre = value(col.pre.heat_duty[0]) / 1e3
    v_top = value(top.flow_mol[0])
    y_nh3 = value(top.mole_frac_comp[0, "NH3"])
    print(f"\n=== vacuum NH3 stripper, {a.n} stages, {a.p} bar "
          f"(T_sat water {t_b - 273.15:.1f} C) ===")
    print(f"feed S13: {value(fs.s13_mass_kgph):.0f} kg/h, NH3 "
          f"{value(fs.nh3_product_kgph):.2f} kg/h")
    print(f"recovery {a.recovery:.3f}: reboiler {q_reb:.1f} kW "
          f"(shortcut {q_short:.1f} kW), feed preheat {q_pre:.1f} kW")
    print(f"overhead {v_top * 18.0 * 3.6:.1f} kg/h-ish, NH3 "
          f"{100 * y_nh3:.1f} mol% (shortcut "
          f"{info['overhead NH3 [mol%]']:.1f} mol%)")
    for k in range(1, a.n + 1):
        fl = getattr(col, f"tray{k}")
        o = fl.control_volume.properties_out[0]
        print(f"  stage {k:2d}: T {value(o.temperature) - 273.15:6.2f} C  "
              f"x_NH3 {value(fl.liq_outlet.mole_frac_comp[0, 'NH3']):.2e}"
              f"  V {value(fl.vap_outlet.flow_mol[0]):.3f} mol/s")


if __name__ == "__main__":
    main()
