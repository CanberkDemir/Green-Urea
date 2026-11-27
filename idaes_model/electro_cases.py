"""
Case comparison: classical (H2 + hydrogenation) vs electrochemical NH3,
both feeding the low-pressure DEF plant; electricity by category and by
unit, pinch analysis, charts, block PFDs, and the JSON used by
optimization/flex_planning.py.

Cases (all at Ft = 8500 kg/h brine, dT_min = 10 K):
  classical      plant_energy case 2: bkp kinetics, B6 to 5000 kg/h, 50 C
                 flash, vacuum stripper + rectifier, plant-wide HI,
                 ambient-source heat pump
  electro        4-stage NO3RR cascade (central), rigorous stripper,
                 plant-wide HI, ambient-source heat pump
  electro+       electro + stripper-overhead MVR + heat pump that lifts
                 below-pinch waste heat (instead of ambient air) to the
                 hot-utility temperature

    python electro_cases.py            # ~3 min; writes figures/electro_*
"""

import contextlib
import io
import json
import logging
import sys
from pathlib import Path

from pyomo.environ import value

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ammonia_energy as AE  # noqa: E402
import electro_flowsheet as EF  # noqa: E402
import plant_energy as PE  # noqa: E402
from heat_integration import pinch, plot_gcc  # noqa: E402

FIG = HERE.parent / "figures"
DT = 10.0
COP_CHILL = 4.0


def hp_cop(t_sink_c, t_source_c):
    th, ts = t_sink_c + 273.15, t_source_c + 273.15
    return max(1.0, AE.HP_CARNOT_FRAC * th / max(th - ts, 1.0))


T_HP_MAX = 160.0   # C, highest sink temperature for an industrial heat pump


def utilities(streams, cop_chill, pinch_hp=False):
    """kWe for heating / cooling after heat integration.

    Hot utility above T_HP_MAX (rectifier reboiler at 198 C, the 190 C
    urea reactor) is electric resistance / e-boiler (COP 1); the rest is a
    heat pump delivering at <= T_HP_MAX. Its source is ambient air (25 C)
    by default; with pinch_hp it is process heat rejected just below the
    pinch, used only if enough such heat is left after the chilled load."""
    res = pinch(streams, DT)
    q_hot, q_cold = res.q_hot_min, res.q_cold_min
    q_chill = res.cold_below(AE.T_CW + 5.0)
    q_e = res.hot_above(T_HP_MAX - DT)          # above the HP limit
    q_hp = max(q_hot - q_e, 0.0)
    t_sink = min(max((s.t_out for s in streams if s.kind == "cold"),
                     default=AE.T_AMB) + DT, T_HP_MAX)
    t_src = AE.T_AMB
    if pinch_hp and q_hp > 0:
        t_src_p = res.t_pinch_shifted - DT / 2 - 5.0
        cop_p = hp_cop(t_sink, t_src_p)
        if q_cold - q_chill >= q_hp * (1 - 1 / cop_p):
            t_src = t_src_p
    cop = hp_cop(t_sink, t_src)
    drawn = q_hp * (1 - 1 / cop) if t_src != AE.T_AMB else 0.0
    q_cw = max(q_cold - q_chill - drawn, 0.0)
    return res, {
        "heat pump": q_hp / cop,
        "e-boiler (>160 C)": q_e,
        "chiller": q_chill / cop_chill,
        "cooling water": AE.CW_ELEC * q_cw,
    }, {"hot utility [kW]": q_hot, "of which >160 C [kW]": q_e,
        "HP COP": cop, "HP source [C]": t_src,
        "chilled [kW]": q_chill, "cooling water [kW]": q_cw}


def def_block(nh3_kgph, D):
    fco2 = nh3_kgph / PE.MW_NH3 / 2 * PE.MW_CO2
    return PE.def_section(D, fco2)


def classical():
    import reactions.ammonia_reactions as AR
    AR.NETWORK = "bkp"
    m = AE.solve_ammonia(fh2=10.0, t_flash_c=50.0, s16_water=5000)
    fs = m.fs
    nh3 = value(fs.nh3_product_kgph)
    st = AE.base_streams(m)
    rs, r_shaft, _ = AE.recovery_streams(
        "S2", nh3, value(fs.h2o_product_kgph), value(fs.s13_mass_kgph),
        value(fs.s13.properties[0].temperature) - 273.15)
    st += rs
    h2 = value(fs.h2_feed.outlet.flow_mol[0]) * 2.016e-3 * 3600
    shaft = {"water electrolysis": h2 * PE.E_SPEC_H2 / PE.FE,
             "H2 compression": value(fs.W_comp) / 1e3,
             "B6 brine concentration":
                 AE.brine_concentration(m)["B6 electricity [kW]"]}
    if r_shaft:
        shaft["NH3 recovery shaft"] = r_shaft
    return nh3, st, shaft, value(fs.refrigeration.cop), {}


def electro(mvr, case="central", rect_p=15.5):
    m = EF.solve(EF.build(case))
    st, _, out = EF.results(m, mvr=mvr, rect_p=rect_p)
    shaft = {"NO3RR cells": out["cell power [kW]"]}
    if mvr:
        shaft["stripper overhead MVR"] = out["overhead MVR [kW]"]
    return out["NH3 product [kg/h]"], st, shaft, COP_CHILL, out


def run(label, a_fun, D, pinch_hp=False):
    nh3, a_st, a_shaft, cop_ch, info = a_fun()
    u_st, u_shaft_kw, urea, u_info = def_block(nh3, D)
    u_shaft = {"DEF compression + pumps": u_shaft_kw}
    shaft = {**a_shaft, **u_shaft}
    res_a, el_a, _ = utilities(a_st, cop_ch, pinch_hp)
    res_u, el_u, _ = utilities(u_st, cop_ch, pinch_hp)
    res_p, el_p, ut_p = utilities(a_st + u_st, cop_ch, pinch_hp)
    tot_sep = sum(a_shaft.values()) + sum(el_a.values()) \
        + u_shaft_kw + sum(el_u.values())
    tot = sum(shaft.values()) + sum(el_p.values())
    r = {
        "label": label, "NH3 [kg/h]": nh3, "urea [kg/h]": urea,
        "DEF [kg/h]": u_info["DEF [kg/h]"],
        "electricity by category [kW]": {**shaft, **el_p},
        "ammonia kW": sum(a_shaft.values()) + sum(el_a.values()),
        "urea kW standalone": u_shaft_kw + sum(el_u.values()),
        "plant kW (HI)": tot,
        "HI credit kW": tot_sep - tot,
        "kWh/kg urea": tot / urea, "kWh/kg NH3": tot / nh3,
        "utilities": ut_p, "info": {**info, **u_info},
        "cell area m2": info.get("cell area [m2]", 0.0),
        "streams": [(s.name, s.section, s.t_in, s.t_out, s.duty_kw)
                    for s in a_st + u_st],
    }
    print(f"\n=== {label}: {nh3:.2f} kg/h NH3 -> {urea:.2f} kg/h urea, "
          f"{tot:.0f} kW = {tot / urea:.1f} kWh/kg urea")
    for k, v in r["electricity by category [kW]"].items():
        print(f"   {k:28s} {v:8.1f} kW  {v / urea:6.2f} kWh/kg urea")
    print("   plant: " + res_p.summary())
    return r, (res_a, res_u, res_p)


def charts(results, pinches):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    cats = []
    for r in results:
        for k in r["electricity by category [kW]"]:
            if k not in cats:
                cats.append(k)
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(results))
    bottom = [0.0] * len(results)
    for i, c in enumerate(cats):
        vals = [r["electricity by category [kW]"].get(c, 0.0) / r["urea [kg/h]"]
                for r in results]
        ax.bar(x, vals, bottom=bottom, color=colors[i % 20], label=c,
               edgecolor="white", width=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for xi, b in zip(x, bottom):
        ax.text(xi, b + 0.4, f"{b:.1f}", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_xticks(list(x), [r["label"] for r in results])
    ax.set_ylabel("Electricity [kWh per kg urea]")
    ax.set_title("Electricity use by category (plant-wide heat integration)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "electro_energy_by_category.png", dpi=200)
    plt.close(fig)

    # per-unit thermal duties, best case
    r = min(results, key=lambda x: x["kWh/kg urea"])   # best case
    st = sorted(r["streams"], key=lambda s: (s[1], -s[4]))
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(st) + 1.5))
    secs = sorted({s[1] for s in st})
    scol = {s: colors[2 * i % 20] for i, s in enumerate(secs)}
    for i, (nm, sec, ti, to, q) in enumerate(st):
        sign = 1 if to > ti else -1
        ax.barh(i, sign * q, color=scol[sec])
        ax.text(sign * q + sign * 5, i, f"{ti:.0f}->{to:.0f} C", va="center",
                ha="left" if sign > 0 else "right", fontsize=7)
    ax.set_yticks(range(len(st)), [s[0] for s in st], fontsize=7)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Duty [kW]  (+ needs heating, - needs cooling)")
    ax.set_title(f"Unit thermal duties before integration: {r['label']}")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=scol[s], label=s) for s in secs],
              fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "electro_unit_duties.png", dpi=200)
    plt.close(fig)

    for r, (ra, ru, rp) in zip(results, pinches):
        tag = r["label"].replace("+", "_plus").replace(" ", "_")
        plot_gcc(rp, FIG / f"electro_gcc_{tag}_plant.png",
                 f"Plant-wide GCC: {r['label']}")
    plot_gcc(pinches[-1][1], FIG / "electro_gcc_DEF.png",
             "DEF section alone (urea side)")


def block_pfd(results):
    import graphviz
    for r in results:
        kw = r["electricity by category [kW]"]
        g = graphviz.Digraph(format="png")
        g.attr(rankdir="LR", fontname="Helvetica", labelloc="t",
               label=f"{r['label']}: {r['NH3 [kg/h]']:.1f} kg/h NH3 -> "
               f"{r['DEF [kg/h]']:.0f} kg/h DEF 32.5 wt% "
               f"({r['kWh/kg urea']:.1f} kWh/kg urea)", fontsize="16")
        g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
               fontsize="10", fillcolor="#eef3fa", color="#557")
        g.node("brine", "Nitrate brine\n8500 kg/h", fillcolor="#d7ecd9")
        g.node("wind", "Wind / battery\nelectricity", fillcolor="#fff3c4")
        if "water electrolysis" in kw:
            g.node("el", f"Water electrolysis\n{kw['water electrolysis']:.0f} kW")
            g.node("cmp", f"H2 compression 27 bar\n{kw['H2 compression']:.0f} kW")
            g.node("b6", f"B6 brine concentration\n{kw['B6 brine concentration']:.0f} kW")
            g.node("rx", "Hydrogenation PFR\n60 C, 27 bar + flash")
            g.edges([("wind", "el"), ("el", "cmp"), ("cmp", "rx"),
                     ("brine", "b6"), ("b6", "rx")])
            last = "rx"
        else:
            g.node("pre", "Feed/effluent HX\n25 -> 65 C")
            g.node("cell", f"NO3RR cascade (4 stages)\n"
                   f"{r['info'].get('cell area [m2]', 0):.0f} m2, "
                   f"FE {r['info'].get('cell FE', 0):.2f}\n"
                   f"{kw['NO3RR cells']:.0f} kW")
            g.node("gls", f"H2 degassing\n{r['info'].get('HER H2 [kg/h]', 0):.1f} kg/h H2")
            g.edges([("brine", "pre"), ("pre", "cell"), ("wind", "cell"),
                     ("cell", "gls")])
            last = "gls"
        mv = kw.get("stripper overhead MVR")
        g.node("str", "Vacuum NH3 stripper\n0.08 bar, 10 stages"
               + (f"\n+ overhead MVR {mv:.0f} kW" if mv else ""))
        g.node("rect", "Rectifier 15 bar\n-> anhydrous NH3")
        g.node("tank", "NH3 buffer tank", shape="cylinder",
               fillcolor="#dbe7f5")
        g.node("def", f"Low-P DEF plant (150 bar reactor,\nMP/LP dissociation, "
               f"stripper)\n{kw['DEF compression + pumps']:.1f} kW shaft")
        g.node("co2", f"CO2\n{r['urea [kg/h]'] * 44.01 / 60.055:.1f} kg/h",
               fillcolor="#d7ecd9")
        g.node("prod", f"DEF 32.5 wt%\n{r['DEF [kg/h]']:.0f} kg/h",
               fillcolor="#f6d6d6")
        g.node("hi", "Heat integration\n(pinch + heat pump)\n"
               f"hot {r['utilities']['hot utility [kW]']:.0f} kW, "
               f"COP {r['utilities']['HP COP']:.1f}",
               fillcolor="#fbe8c8", shape="ellipse")
        g.edges([(last, "str"), ("str", "rect"), ("rect", "tank"),
                 ("tank", "def"), ("co2", "def"), ("def", "prod")])
        g.edge("hi", "str", style="dashed", color="#b5402a")
        g.edge("hi", "def", style="dashed", color="#b5402a")
        tag = r["label"].replace("+", "_plus").replace(" ", "_")
        g.attr(dpi="170")
        g.render(FIG / f"block_pfd_{tag}", cleanup=True)


def main():
    logging.disable(logging.WARNING)
    D = PE.load_def_module(None)
    cases = [("classical", classical, False),
             ("electro", lambda: electro(False), False),
             ("electro+", lambda: electro(True), True),
             ("electro++", lambda: electro(True, rect_p=6.2), True)]
    results, pinches = [], []
    for label, fun, php in cases:
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            r, p = run(label, fun, D, pinch_hp=php)
        print(buf.getvalue().split("\n=== ")[-1] if "=== " in buf.getvalue()
              else buf.getvalue())
        results.append(r)
        pinches.append(p)
    # parameter range on the electro+ configuration (totals only)
    rng = {}
    for c in ("optimistic", "pessimistic"):
        with contextlib.redirect_stdout(io.StringIO()):
            r, _ = run(f"electro+ {c}", lambda c=c: electro(True, c), D, True)
        print(f"electro+ {c}: {r['kWh/kg urea']:.1f} kWh/kg urea")
        r.pop("streams")
        rng[f"electro_plus_{c}"] = r
    charts(results, pinches)
    block_pfd(results)
    js = {"classical": results[0], "electro_central": results[1],
          "electro_plus_central": results[2],
          "electro_plusplus_central": results[3]}
    for v in js.values():
        v.pop("streams", None)
    js.update(rng)
    (FIG / "electro_cases.json").write_text(json.dumps(js, indent=1))
    print(f"wrote figures/electro_* and {FIG / 'electro_cases.json'}")


if __name__ == "__main__":
    main()
