"""
Rigorous section B (NH3 recovery) at the IPPS operating point: rigorous
vacuum stripper with a pressure-drop profile + IDAES MVR compressor on the
overhead + shortcut 15.5 bar rectifier. Checks the shortcut optimum of
recovery_opt.py (0.06 bar, 20 stages, MVR COP 8 -> 30.3 kW).

  * feed: S13 of the IDAES ammonia flowsheet (base case), NH3 scaled to the
    IPPS operating point (default 9.85 kg/h aq. NH3);
  * stripper: nh3_stripper.build_column (MESH stages, ammonia two-phase
    package, no salting-out), stage k at p_top + (k - 1) * dp, reboiler
    duty solved for 99 % NH3 recovery;
  * MVR: IDAES PressureChanger (isentropic, eta 0.75) on IAPWS-95 steam,
    the overhead taken as water vapour (it is ~95 mol% water; the NH3 lowers
    the dew point a little, so the condensing T is taken 2 K below Tsat).
    The outlet pressure is the lowest at which the overhead condenses
    >= DT_MVR above the bottom-stage (reboiler) temperature; the condensing
    overhead supplies the reboiler, any shortfall goes to the heat pump;
  * rectifier + utilities: electro_flowsheet.rectifier_streams and the
    same pinch/heat-pump/chiller costing as idaes_data.elec().

    python recovery_rigorous.py                 # sweep, writes CSV
    python recovery_rigorous.py --p 0.1 --n 15 --dp 1.0
"""

import argparse
import contextlib
import io
import logging
import math
import sys
from pathlib import Path

import pandas as pd
from pyomo.environ import Block, ConcreteModel, Constraint, value

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "surrogates"))
logging.disable(logging.WARNING)

import ammonia_energy as AE  # noqa: E402
import nh3_stripper as NS  # noqa: E402
from heat_integration import Stream  # noqa: E402
from idaes.core import FlowsheetBlock  # noqa: E402
from idaes.core.solvers import get_solver  # noqa: E402

DT_MVR = 10.0      # K, condensing overhead above the reboiler
ETA_COMP = 0.75    # isentropic efficiency of the vapour compressor
NH3_DEW_DEPRESSION = 2.0  # K, overhead condenses below water Tsat
RECOVERY = 0.99
OUT = HERE.parent / "figures" / "recovery_rigorous.csv"


class _State:
    def __init__(self, flow, x, t, p):
        self.flow_mol, self.mole_frac_comp = flow, x
        self.temperature, self.pressure = t, p


def feed_state(nh3_kgph):
    m = AE.solve_ammonia(fh2=10.0, t_flash_c=50.0, s16_water=5000.0)
    s = m.fs.s13.properties[0]
    n = {j: value(s.flow_mol * s.mole_frac_comp[j]) for j in s.mole_frac_comp}
    n["NH3"] = nh3_kgph / 3600.0 / 17.031e-3
    tot = sum(n.values())
    st = _State(tot, {j: v / tot for j, v in n.items()},
                value(s.temperature), value(s.pressure))
    w_kgph = n["H2O"] * 18.015e-3 * 3600
    s13_kgph = value(m.fs.s13_mass_kgph) * tot / value(s.flow_mol)
    return m, st, w_kgph, s13_kgph


def mvr_work_kw(n_vap, t_in, p_in, p_out):
    """IDAES isentropic compressor on IAPWS-95 steam [kW]."""
    from idaes.models.properties import iapws95
    from idaes.models.unit_models import PressureChanger
    from idaes.models.unit_models.pressure_changer import \
        ThermodynamicAssumption
    mc = ConcreteModel()
    mc.fs = FlowsheetBlock(dynamic=False)
    mc.fs.w = iapws95.Iapws95ParameterBlock()
    mc.fs.c = PressureChanger(property_package=mc.fs.w, compressor=True,
                              thermodynamic_assumption=ThermodynamicAssumption.isentropic)
    h = iapws95.htpx(T=max(t_in, 0.5 + 273.15) * __import__("pyomo.environ").environ.units.K,
                     P=p_in * __import__("pyomo.environ").environ.units.Pa)
    mc.fs.c.inlet.flow_mol.fix(n_vap)
    mc.fs.c.inlet.enth_mol.fix(h)
    mc.fs.c.inlet.pressure.fix(p_in)
    mc.fs.c.outlet.pressure.fix(p_out)
    mc.fs.c.efficiency_isentropic.fix(ETA_COMP)
    with contextlib.redirect_stdout(io.StringIO()):
        mc.fs.c.initialize(outlvl=50)
        r = get_solver("ipopt").solve(mc)
    assert str(r.solver.termination_condition) == "optimal"
    return value(mc.fs.c.work_mechanical[0]) / 1e3, \
        value(mc.fs.c.control_volume.properties_out[0].temperature)


def design(m, st, w_kgph, s13_kgph, p_top, n, dp_mbar, nh3_kgph):
    from idaes_data import elec
    from electro_flowsheet import rectifier_streams
    if hasattr(m.fs, "strip"):
        m.fs.del_component(m.fs.strip)
    p_pa = p_top * 1e5
    t_top = NS.water_tsat(p_top)
    col = NS.build_column(m, m.fs.props_vle, st, p_pa, n, t_top - 2.0)
    for k in range(1, n + 1):           # pressure-drop profile
        pk = p_pa + (k - 1) * dp_mbar * 100.0
        getattr(col, f"mix{k}").mixed_state[0].pressure.fix(pk)
    col.pre.control_volume.properties_out[0].pressure.fix(p_pa)
    solver = get_solver("ipopt", options=NS._OPT)
    rs, _, _ = AE.recovery_streams("S2", nh3_kgph, w_kgph, s13_kgph,
                                   value(st.temperature) - 273.15)
    q0 = [s for s in rs if s.name == "stripper reboiler"][0].duty_kw
    with contextlib.redirect_stdout(io.StringIO()):
        NS.initialize_column(col, n, q0 * 1e3, solver)
    for blk in col.component_data_objects(Block, descend_into=True):
        if hasattr(blk, "eps_1_Vap_Liq"):
            blk.eps_1_Vap_Liq = 1.0
    reb = getattr(col, f"tray{n}")
    solver.solve(col)
    nh3_in = st.flow_mol * st.mole_frac_comp["NH3"]
    bot = reb.liq_outlet
    col.recovery_spec = Constraint(
        expr=bot.flow_mol[0] * bot.mole_frac_comp[0, "NH3"]
        == (1 - RECOVERY) * nh3_in)
    reb.heat_duty.unfix()
    for _ in range(3):
        r = solver.solve(col)
        if str(r.solver.termination_condition) == "optimal":
            break
    if str(r.solver.termination_condition) != "optimal":
        return None
    top = col.tray1.vap_outlet
    v = value(top.flow_mol[0])
    y = value(top.mole_frac_comp[0, "NH3"])
    t_top_s = value(top.temperature[0])
    t_bot = value(reb.control_volume.properties_out[0].temperature)
    q_reb = value(reb.heat_duty[0]) / 1e3
    q_pre = value(col.pre.heat_duty[0]) / 1e3
    # MVR: lowest outlet P whose (dew-depressed) Tsat >= t_bot + DT_MVR
    t_need = t_bot + DT_MVR + NH3_DEW_DEPRESSION
    p_out = _psat_pa(t_need)
    w_mvr, t_dis = mvr_work_kw(v, t_top_s, p_pa, p_out)
    q_cond_ov = q_reb + w_mvr * 0.0   # latent ~ reboiler duty (same vapour)
    # streams left after the MVR loop: reboiler deficit/surplus, overhead
    # condensate subcooling to 25 C, feed preheat, bottoms cooling, rectifier
    t_c = t_need - NH3_DEW_DEPRESSION
    w_ov = v * (1 - y) * 3.6          # kmol/h water in overhead
    st_list = []
    if q_pre > 1e-3:
        st_list.append(Stream("stripper feed preheat",
                              value(st.temperature) - 273.15, t_top - 275.15,
                              q_pre, section="B"))
    surplus = q_cond_ov + w_mvr - q_reb   # superheat/work ends up as heat
    if surplus > 1e-3:
        st_list.append(Stream("MVR surplus condensing", t_c - 273.15 + 0.1,
                              t_c - 273.15, surplus, section="B"))
    st_list.append(Stream("overhead condensate cooler", t_c - 273.15, 25.0,
                          v * 3.6 * 18.0 * 4.2 * (t_c - 298.15) / 3600,
                          section="B"))
    st_list.append(Stream("stripper bottoms cooler", t_bot - 273.15,
                          AE.T_CW + 5.0,
                          s13_kgph * AE.CP_BRINE * (t_bot - 273.15 - AE.T_CW
                                                    - 5.0) / 3600,
                          section="B"))
    rect, rinfo = rectifier_streams(RECOVERY * nh3_kgph, w_ov)
    e_total = elec(st_list + rect, w_mvr, 4.0)
    return {"p_top [bar]": p_top, "stages": n, "dp [mbar/stage]": dp_mbar,
            "p_bottom [bar]": (p_pa + (n - 1) * dp_mbar * 100) / 1e5,
            "T_top [C]": t_top_s - 273.15, "T_bottom [C]": t_bot - 273.15,
            "reboiler [kW]": q_reb, "overhead [kg/h]": v * 3.6 * 18.0,
            "overhead NH3 [mol%]": 100 * y,
            "overhead vol. flow [m3/s]": v * 8.314 * t_top_s / p_pa,
            "MVR p_out [bar]": p_out / 1e5, "MVR ratio": p_out / p_pa,
            "MVR work [kW]": w_mvr, "MVR discharge T [C]": t_dis - 273.15,
            "B electricity [kW]": e_total,
            "kWh/kg NH3": e_total / (RECOVERY * nh3_kgph),
            "rectifier R": rinfo["rectifier R"]}


def _psat_pa(t_k):
    a, b, c = 4.6543, 1435.264, -64.848     # Antoine (NIST), as NS
    return 10 ** (a - b / (t_k + c)) * 1e5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nh3", type=float, default=9.85)
    ap.add_argument("--p", type=float, nargs="*",
                    default=[0.06, 0.08, 0.10, 0.15, 0.20])
    ap.add_argument("--n", type=int, nargs="*", default=[10, 15, 20])
    ap.add_argument("--dp", type=float, nargs="*", default=[1.0, 2.0])
    a = ap.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        m, st, w, s13 = feed_state(a.nh3)
    print(f"feed: NH3 {a.nh3} kg/h, water {w:.0f} kg/h, S13 {s13:.0f} kg/h")
    rows = []
    for p in a.p:
        for n in a.n:
            for dp in a.dp:
                try:
                    r = design(m, st, w, s13, p, n, dp, a.nh3)
                except Exception as e:  # noqa: BLE001
                    r = None
                    print(f"  p {p} n {n} dp {dp}: error {e}")
                if r is None:
                    print(f"  p {p} n {n} dp {dp}: not converged")
                    continue
                rows.append(r)
                print(f"  p {p:.2f}-{r['p_bottom [bar]']:.3f} bar n {n:2d} "
                      f"dp {dp}: T {r['T_top [C]']:.1f}-{r['T_bottom [C]']:.1f} C"
                      f" reb {r['reboiler [kW]']:.0f} kW, MVR {r['MVR work [kW]']:.1f}"
                      f" kW (x{r['MVR ratio']:.2f}, {r['overhead vol. flow [m3/s]']:.2f}"
                      f" m3/s) -> B {r['B electricity [kW]']:.1f} kW", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    if len(df):
        b = df.loc[df["B electricity [kW]"].idxmin()]
        print("\nbest:\n" + b.to_string())
        memb = a.nh3 * 14.007 / 17.031 * AE.MEMBRANE_KWH_PER_KG_N
        print(f"\nmembrane black box (7.1 kWh/kg N): {memb:.1f} kW")


if __name__ == "__main__":
    main()
