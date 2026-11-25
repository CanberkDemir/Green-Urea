"""Draw a process flow diagram of the IDAES urea (or ammonia) flowsheet.

The diagram is generated from the model's own Arc connectivity, so it
always matches what IDAES actually solves. Units are grouped by plant
section and drawn with a shape per unit type; vapor/liquid outlets of
flashes are tagged V/L.

    python draw_pfd.py              # urea -> figures/urea_pfd.{svg,png}
    python draw_pfd.py ammonia      # ammonia -> figures/ammonia_pfd.{svg,png}
"""

import sys
from pathlib import Path

import graphviz
from pyomo.network import Arc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# IDAES unit class -> (shape, fill colour)
STYLE = {
    "Feed": ("cds", "#d7ecd9"),
    "Product": ("cds", "#f6d6d6"),
    "Mixer": ("circle", "#eeeeee"),
    "Separator": ("triangle", "#eeeeee"),
    "Flash": ("cylinder", "#dbe7f5"),
    "Heater": ("box", "#fbe8c8"),
    "Pump": ("invtriangle", "#e3dcf2"),
    "Compressor": ("trapezium", "#e3dcf2"),
    "PressureChanger": ("invtriangle", "#e3dcf2"),
    "CSTR": ("box3d", "#f7cfa3"),
    "PFR": ("box3d", "#f7cfa3"),
    "StoichiometricReactor": ("box3d", "#f2dfc4"),
    "Translator": ("point", "#999999"),
}

UREA_SECTIONS = {
    "Feeds": ["nh3_feed", "co2_feed", "steam_feed"],
    "HP synthesis loop (138 bar, 200 C)": [
        "b5", "nh3p1", "nh3sp1", "nh3p2", "b6", "m01", "b27", "r01", "b13"],
    "MP decomposition / recycle (20 bar)": [
        "b2", "f01", "a02", "b1", "a01", "b11", "b15", "b10h", "b12", "b9",
        "b7", "b7_mix", "b19", "b16h", "b17"],
    "LP decomposition (4 bar)": [
        "b22", "f012", "a022", "carbmix2", "b8", "a012", "b3", "b18"],
    "Finishing & stripping (5 -> 1 bar)": [
        "x3", "b4", "a013", "b24", "b20h", "p1", "b25_mix", "b25", "b25col",
        "b23", "b14", "b26", "b28"],
    "Products / purges": [
        "def_prod", "s30_prod", "s42_prod", "purge1_prod", "purge2_prod",
        "purge3_prod"],
}

AMMONIA_SECTIONS = {
    "Feeds": ["h2_feed", "no3_feed"],
    "H2 compression (4 stages, intercooled to 50 C, 27 bar)": [
        "cmp1", "ic1", "cmp2", "ic2", "cmp3", "ic3", "cmp4"],
    "Brine pre-concentration": ["b6"],
    "Reaction (27 bar)": ["mixer1", "h1", "b1"],
    "Separation & let-down": ["c1", "b2", "split1", "v1"],
    "Products / purges": ["h2o_out", "s13", "purge"],
}

DEF_SECTIONS = {
    "Feeds": ["co2_feed", "nh3_feed", "steam_feed", "water_feed"],
    "HP synthesis (150 bar, 190 C) - reactor only": [
        "nh3_pump", "rec_pump", "m_r", "reactor"],
    "MP dissociation (20 bar, 160 C)": ["v_mp", "d_mp", "f_mp"],
    "LP dissociation (4 bar, 135 C)": ["v_lp", "d_lp", "f_lp"],
    "Purification (0.5 bar steam stripping) & DEF finishing": [
        "v_pu", "v_steam", "m_strip", "stripper", "def_cool", "m_def",
        "def_trim"],
    "Condensate handling": ["vac_cond", "cond_pump", "cond_split", "nh3s",
                            "v_cond_def"],
    "Carbamate recovery": ["m_lpc", "lpc", "f_lpc", "lp_pump", "lp_comp",
                           "m_mpc", "mpc", "nh3_cond", "f_mpc"],
    "Products / vents": ["def_prod", "vent_mp", "vent_vac"],
}

# (source, destination) pairs drawn as recycle streams
RECYCLES = {("b26", "m01"), ("b9", "b13"), ("b17", "b5"), ("b7", "b15"),
            ("split1", "mixer1"), ("rec_pump", "m_r"),
            ("f_mpc", "rec_pump"), ("lp_pump", "m_mpc"),
            ("lp_comp", "m_mpc"), ("cond_split", "m_lpc"),
            ("nh3s", "m_lpc")}


# per-flowsheet labels that would otherwise collide between models
LABELS = {
    "urea": {"steam_feed": "STRIPH2O", "def_prod": "DEF-OUT", "b1": "B1"},
    "ammonia": {"b1": "B1 (RPlug)"},
    "def": {"steam_feed": "LP steam", "def_prod": "DEF 32.5 wt%"},
}
_ACTIVE = {}


def _label(name):
    if name in _ACTIVE:
        return _ACTIVE[name]
    nice = {"b10h": "B10", "b16h": "B16", "b20h": "B20", "b7_mix": "B7 feed",
            "b25_mix": "B25 feed", "nh3_feed": "NH3-IN", "co2_feed": "CO2-IN",
            "steam_feed": "STRIPH2O", "def_prod": "DEF-OUT", "s30_prod": "S30",
            "s42_prod": "S42", "purge1_prod": "PURGE1",
            "purge2_prod": "PURGE2", "purge3_prod": "PURGE3",
            "b25col": "B25 (10 trays)",
            "h2_feed": "H2FEED", "no3_feed": "NO3-IN", "h2o_out": "H2O-OUT",
            "s13": "S13 (NH3 product)", "purge": "PURGE1",
            "mixer1": "MIXER1", "split1": "SPLIT1",
            "cmp1": "MCOMPR1 stage 1", "cmp2": "stage 2",
            "cmp3": "stage 3", "cmp4": "stage 4", "ic1": "IC1",
            "ic2": "IC2", "ic3": "IC3", "b1": "B1 (RPlug)",
            "co2_feed": "CO2", "water_feed": "Demin water",
            "steam_feed": "LP steam", "def_prod": "DEF 32.5 wt%",
            "vent_mp": "MP vent", "vent_vac": "Vacuum vent",
            "nh3_pump": "NH3 HP pump", "rec_pump": "Carbamate HP pump",
            "m_r": "M-R", "reactor": "Urea reactor",
            "v_mp": "Valve to 20 bar", "d_mp": "MP decomposer",
            "f_mp": "MP separator", "v_lp": "Valve to 4 bar",
            "d_lp": "LP decomposer", "f_lp": "LP separator",
            "v_pu": "Valve to 0.5 bar", "v_steam": "Steam valve",
            "m_strip": "M-strip", "stripper": "Vacuum steam stripper",
            "def_cool": "Solution cooler", "m_def": "DEF dilution",
            "def_trim": "DEF trim cooler", "vac_cond": "Vacuum condenser",
            "cond_pump": "Condensate pump", "cond_split": "Condensate split",
            "nh3s": "Condensate NH3 stripper",
            "v_cond_def": "Stripped-water cooler", "m_lpc": "M-LPC",
            "lpc": "LP carbamate condenser", "f_lpc": "LPC separator",
            "lp_pump": "LP-MP pump", "lp_comp": "LP gas compressor",
            "m_mpc": "M-MPC", "mpc": "MP carbamate condenser",
            "nh3_cond": "NH3 condenser", "f_mpc": "MPC separator"}
    return nice.get(name, name.upper())


def _top_unit(fs, comp):
    """Walk up from a port to the unit directly under fs."""
    blk = comp.parent_block()
    while blk.parent_block() is not fs:
        blk = blk.parent_block()
    return blk


def _phase_tag(port):
    n = port.local_name.lower()
    if n.startswith("vap"):
        return "V"
    if n.startswith("liq"):
        return "L"
    return ""


def build_graph(m, sections, title):
    fs = m.fs
    g = graphviz.Digraph("pfd", format="svg")
    g.attr(rankdir="LR", splines="spline", nodesep="0.35", ranksep="0.55",
           fontname="Helvetica", fontsize="20", labelloc="t", label=title,
           newrank="true")
    g.attr("node", fontname="Helvetica", fontsize="11", style="filled",
           color="#555555", penwidth="1.0")
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#444444",
           arrowsize="0.7")

    edges, units = [], set()
    # top-level arcs only: internal arcs of a sub-flowsheet (B25 trays)
    # are drawn as the single column node
    for arc in fs.component_objects(Arc, descend_into=False):
        src_u = _top_unit(fs, arc.source)
        dst_u = _top_unit(fs, arc.destination)
        if src_u is dst_u:
            continue
        edges.append((src_u, dst_u, _phase_tag(arc.source)))

    # Translators only bridge property packages - draw straight through
    def is_tr(u):
        return "Translator" in type(u).__name__
    changed = True
    while changed:
        changed = False
        for i, (s, d, t) in enumerate(edges):
            if is_tr(d):
                outs = [e for e in edges if e[0] is d]
                edges = [e for e in edges if e[0] is not d
                         and e is not edges[i]]
                edges += [(s, o[1], t or o[2]) for o in outs]
                changed = True
                break
    for s, d, _ in edges:
        units.update([s, d])

    placed = set()
    for i, (sec, names) in enumerate(sections.items()):
        with g.subgraph(name=f"cluster_{i}") as c:
            c.attr(label=sec, style="rounded,dashed", color="#999999",
                   fontsize="13", fontcolor="#333333")
            # strict left-to-right reading: feeds in the first column,
            # products/vents in the last
            if sec.lower().startswith("feed"):
                c.attr(rank="min")
            elif sec.lower().startswith("product"):
                c.attr(rank="max")
            for nm in names:
                u = getattr(fs, nm, None)
                if u is None or u not in units:
                    continue
                cls = type(u).__name__.lstrip("_").replace("Scalar", "")
                shape, fill = STYLE.get(cls, ("box", "#ffffff"))
                if nm == "b25col":
                    shape, fill = "box3d", "#dbe7f5"
                c.node(u.local_name, _label(nm), shape=shape, fillcolor=fill,
                       width="0.5" if shape in ("circle", "triangle",
                                                "invtriangle") else "0.9")
                placed.add(u)
    for u in units - placed:
        cls = type(u).__name__.lstrip("_").replace("Scalar", "")
        shape, fill = STYLE.get(cls, ("box", "#ffffff"))
        g.node(u.local_name, _label(u.local_name), shape=shape,
               fillcolor=fill)

    for s, d, tag in edges:
        rec = (s.local_name, d.local_name) in RECYCLES
        g.edge(s.local_name, d.local_name, label=tag,
               # recycles must not drag the layout backwards
               constraint="false" if rec else "true",
               color="#b5402a" if rec else "#444444")
    return g


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "urea"
    _ACTIVE.update(LABELS.get(which, {}))
    out_dir = HERE.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    if which == "urea":
        import urea_flowsheet as U
        m = U.build_model()
        g = build_graph(m, UREA_SECTIONS,
                        "Green urea: IDAES flowsheet (from aspen/ureaF.bkp)")
    elif which == "def":
        import def_lowp_flowsheet as D
        m = D.build_model()
        g = build_graph(m, DEF_SECTIONS, "Low-pressure DEF urea plant "
                        "(after US 10,759,745 B2): IDAES flowsheet")
    else:
        import ammonia_flowsheet as A
        m = A.build_model() if hasattr(A, "build_model") else A.build()
        g = build_graph(m, AMMONIA_SECTIONS, "Green ammonia: IDAES flowsheet "
                               "(from aspen/ammoniaF.bkp)")
    base = out_dir / f"{which}_pfd"
    for fmt in ("svg", "png"):
        g.format = fmt
        if fmt == "png":
            g.attr(dpi="200")
        g.render(base, cleanup=True)
    print(f"wrote {base}.svg and {base}.png")


if __name__ == "__main__":
    main()
