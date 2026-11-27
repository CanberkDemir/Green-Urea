"""
Hub-and-spoke network for green DEF from nitrate ion-exchange (IX) brines.

Spoke = an IX nitrate-removal works: its spent regenerant brine goes through
the electrochemical route (NO3RR cascade + vacuum stripper/MVR + rectifier,
figures/electro_cases.json 'electro_plus_central') to anhydrous NH3, which is
stored and trucked to ONE hub that runs the low-pressure DEF plant
(def_lowp_flowsheet, optimized design). All sites take power from one
offshore wind farm (PPA at the CfD AR6 price); the cells follow the wind
and the NH3 tanks decouple the spokes from the hub (see ipps_electro.py).

Every input is in ASSUMPTIONS with its source or marked 'assumption'.

    python hub_spoke.py           # base case, sweeps, phased build-out
Writes optimization/solutions/hub_spoke_*.csv/json and figures/hub_spoke_*.png
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "optimization" / "solutions"
FIG = ROOT / "figures"
MW_NO3, MW_NH3, MW_UREA, MW_CO2 = 62.004, 17.031, 60.055, 44.01

# ------------------------------------------------------------- assumptions
A = dict(
    # electrochemical spoke (IDAES, electro_plus_central)
    spoke_kwh_per_kg_nh3=None,       # read from electro_cases.json
    area_m2_per_kgph_nh3=None,       # read from electro_cases.json
    nitrate_to_nh3_yield=0.95,       # cascade conversion (electro_no3rr.py)
    cell_oversize=1.3,               # cells follow wind (flex_planning, ipps_electro)
    stack_gbp_per_m2=4700.0,         # ~$6000/m2 central (electro_no3rr.py)
    stack_life_y=5.0,                # assumption (G4: catalyst life undemonstrated)
    spoke_bop_ref_gbp=1.5e6,         # assumption: stripper+MVR+rectifier+tank+civils at 20 kg/h NH3
    spoke_ref_kgph=20.15,
    scale_exp=0.6,                   # six-tenths rule
    spoke_staff_fte=0.5,             # unmanned, utility staff (assumption)
    # DEF hub (def_lowp, optimized: 0.52 kWh/kg urea at 5 kg/h; 0.68 old design)
    hub_kwh_per_kg_urea=0.6,
    # urea-section capex benchmark: integrated 3,500 t/d urea complex
    # USD 1.6-2.8 bn (credenceresearch); urea share 35 % (assumption);
    # scaled with exponent hub_exp (0.6 base; 0.7-0.8 modular, sensitivity)
    bench_usd=2.2e9, bench_urea_share=0.35, bench_tpy=3500 * 350,
    hub_exp=0.6, usd_to_gbp=0.78,
    hub_staff_fte=4.0, fte_gbp=55e3,  # assumption
    maint_frac=0.04,                 # of capex per year (assumption)
    co2_gbp_per_kg=0.02,             # IPPS c_CO2
    # power: one offshore wind farm, PPA at CfD AR6 fixed offshore strike
    # GBP 58.87/MWh (2012 prices) = ~GBP 82/MWh today (ORE Catapult)
    power_gbp_per_mwh=82.0,
    # transport: anhydrous NH3 ADR tanker. RHA 2026 cost tables: GBP 208k/y
    # incl. fuel for a 44 t artic over ~85,000 mi (136,800 km) = GBP 1.52/km;
    # +30 % ADR/pressure-tanker premium (assumption) = GBP 2.0/km
    truck_gbp_per_km=2.0, truck_fixed_gbp=300.0,   # loading/driver time
    truck_payload_t=10.0,            # small pressure tanker / ISO (assumption)
    road_factor=1.3,                 # road / great-circle (assumption)
    nh3_tank_gbp_per_kg=1.89,        # IPPS c_NH3
    # product and credits
    def_gbp_per_l=0.30,              # bulk AdBlue 26-46p/l (trade press, undated)
    def_density=1.09,
    co2_avoided_t_per_t_nh3=2.4,     # IEA Ammonia Technology Roadmap (grey NH3)
    carbon_gbp_per_t=49.41,          # UK ETS 2026 reference price (GOV.UK)
    # waste: brine tankering avoided (where tankered) and salt reuse
    tanker_disposal_gbp_per_m3=15.0, # utility tankered-waste tariffs GBP 10-25/m3
    salt_gbp_per_t=80.0,             # assumption
    brine_nacl_kg_per_m3=93.5,       # 1.6 M NaCl (model feed)
    salt_reuse_frac=0.5,             # assumption: NO3-free brine reused as regenerant
    brine_g_no3_per_l=9.8,           # model feed (electro_no3rr.py)
    discount=0.07, life_y=20,
)

# hub candidate and spokes. Plant facts from UK Water Projects case studies;
# coordinates are approximate (town level) - assumption for distances.
HUB = dict(name="Saltend Chemicals Park, Hull", lat=53.737, lon=-0.246)
SITES = [
    # name, company, IX Ml/d, raw NO3 mg/l, treated NO3 mg/l, lat, lon, tankered?
    ("Keldgate WTW (Cottingham)", "Yorkshire Water", 33.0, 60.0, 5.0,
     53.785, -0.420, False),
    ("Tophill Low WTW", "Yorkshire Water", 22.0, 64.0, 5.0,
     53.917, -0.371, False),      # waste piped 12 km to Beverley WwTW
    ("Lyng Forge WTW (Norfolk)", "Anglian Water", 6.0, 77.0, 2.5,
     52.719, 1.054, True),
]
# generic further spokes: Anglian Water's AMP4 programme covered 22 sites;
# names/sizes not public in the sources read -> modelled as a typical spoke
GENERIC = dict(ml_d=8.0, no3_in=70.0, no3_out=3.0, tankered=True)
GENERIC_KM = [60, 90, 120, 150, 180, 210, 240, 270, 300]   # assumption


def crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def gc_km(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    return 6371 * math.acos(min(1, math.sin(p1) * math.sin(p2)
                                + math.cos(p1) * math.cos(p2) * math.cos(dl)))


def load_plant():
    d = json.loads((ROOT / "figures" / "electro_cases.json").read_text())
    e = d["electro_plus_central"]
    nh3 = e["NH3 [kg/h]"]
    A["spoke_kwh_per_kg_nh3"] = e["ammonia kW"] / nh3
    A["area_m2_per_kgph_nh3"] = e["info"]["cell area [m2]"] / nh3


def spokes():
    rows = []
    for nm, co, ml, cin, cout, lat, lon, tk in SITES:
        km = A["road_factor"] * gc_km((HUB["lat"], HUB["lon"]), (lat, lon))
        rows.append(dict(name=nm, company=co, ml_d=ml, no3_in=cin,
                         no3_out=cout, road_km=km, tankered=tk, generic=False))
    for i, km in enumerate(GENERIC_KM):
        rows.append(dict(name=f"typical IX works #{i + 1}", company="(generic)",
                         ml_d=GENERIC["ml_d"], no3_in=GENERIC["no3_in"],
                         no3_out=GENERIC["no3_out"], road_km=km,
                         tankered=GENERIC["tankered"], generic=True))
    df = pd.DataFrame(rows)
    no3_kgd = df.ml_d * 1e6 * (df.no3_in - df.no3_out) / 1e6     # kg/d
    df["no3_kg_d"] = no3_kgd
    df["nh3_kg_h"] = no3_kgd * A["nitrate_to_nh3_yield"] * MW_NH3 / MW_NO3 / 24
    df["brine_m3_d"] = no3_kgd / A["brine_g_no3_per_l"]
    df["brine_pct_of_flow"] = 100 * df.brine_m3_d / (df.ml_d * 1e3)
    return df


def spoke_annual(s):
    """Annualised cost (GBP/y) and outputs of one spoke (no DEF)."""
    r, n = A["discount"], A["life_y"]
    nh3_y = s.nh3_kg_h * 8760                                   # kg/y
    area = A["area_m2_per_kgph_nh3"] * s.nh3_kg_h * A["cell_oversize"]
    stack = area * A["stack_gbp_per_m2"]
    bop = A["spoke_bop_ref_gbp"] * (s.nh3_kg_h / A["spoke_ref_kgph"]) \
        ** A["scale_exp"]
    tank = A["nh3_tank_gbp_per_kg"] * 2 * A["truck_payload_t"] * 1e3
    capex = bop + tank + stack
    ann_capex = (bop + tank) * crf(r, n) + stack * crf(r, A["stack_life_y"])
    power = nh3_y * A["spoke_kwh_per_kg_nh3"] / 1e3 * A["power_gbp_per_mwh"]
    om = A["maint_frac"] * (bop + tank) + A["spoke_staff_fte"] * A["fte_gbp"]
    return dict(capex=capex, ann_capex=ann_capex, power=power, om=om,
                nh3_y=nh3_y)


def transport_annual(s, to_hub=True):
    if not to_hub:
        return 0.0, 0.0
    trips = s.nh3_kg_h * 8760 / (A["truck_payload_t"] * 1e3)
    cost = trips * (2 * s.road_km * A["truck_gbp_per_km"] + A["truck_fixed_gbp"])
    return cost, trips


def def_plant_capex(urea_tpy):
    usd = A["bench_usd"] * A["bench_urea_share"] \
        * (urea_tpy / A["bench_tpy"]) ** A["hub_exp"]
    return usd * A["usd_to_gbp"]


def def_annual(nh3_y_kg, capacity_tpy=None, staff=None):
    """DEF plant processing nh3_y_kg (kg NH3/y), built for capacity_tpy."""
    r, n = A["discount"], A["life_y"]
    urea = nh3_y_kg * MW_UREA / (2 * MW_NH3)                   # kg/y
    cap = capacity_tpy if capacity_tpy else urea / 1e3
    capex = def_plant_capex(cap)
    co2 = urea * MW_CO2 / MW_UREA * A["co2_gbp_per_kg"]
    power = urea * A["hub_kwh_per_kg_urea"] / 1e3 * A["power_gbp_per_mwh"]
    om = A["maint_frac"] * capex + (staff if staff is not None
                                    else A["hub_staff_fte"]) * A["fte_gbp"]
    return dict(capex=capex, ann_capex=capex * crf(r, n), co2=co2,
                power=power, om=om, urea=urea)


def credits(s, nh3_y):
    carbon = nh3_y / 1e3 * A["co2_avoided_t_per_t_nh3"] * A["carbon_gbp_per_t"]
    brine_y = s.brine_m3_d * 365
    # the denitrified brine is still ~1.6 M NaCl: only the reused share
    # avoids both its disposal (tankering, where tankered) and fresh salt
    reused = brine_y * A["salt_reuse_frac"]
    waste = reused * ((A["tanker_disposal_gbp_per_m3"] if s.tankered else 0.0)
                      + A["brine_nacl_kg_per_m3"] / 1e3 * A["salt_gbp_per_t"])
    return carbon, waste


def network(df, n_spokes, mode="hub"):
    """mode 'hub': first n spokes (sorted by distance) feed one hub.
    mode 'local': each spoke runs its own DEF plant (no transport)."""
    sel = df.sort_values("road_km").head(n_spokes)
    tot = dict(ann_capex=0, power=0, om=0, transport=0, co2=0, carbon=0,
               waste=0, urea=0, capex=0, trips=0)
    nh3_hub = 0.0
    for _, s in sel.iterrows():
        sp = spoke_annual(s)
        for k in ("ann_capex", "power", "om", "capex"):
            tot[k] += sp[k]
        c, w = credits(s, sp["nh3_y"])
        tot["carbon"] += c
        tot["waste"] += w
        if mode == "hub":
            t, trips = transport_annual(s)
            tot["transport"] += t
            tot["trips"] += trips
            nh3_hub += sp["nh3_y"]
        else:
            d = def_annual(sp["nh3_y"], staff=1.0)
            for k in ("ann_capex", "power", "om", "co2", "capex"):
                tot[k] += d[k]
            tot["urea"] += d["urea"]
    if mode == "hub":
        d = def_annual(nh3_hub)
        for k in ("ann_capex", "power", "om", "co2", "capex"):
            tot[k] += d[k]
        tot["urea"] += d["urea"]
    urea = tot["urea"]
    def_kg = urea / 0.325
    revenue = def_kg / A["def_density"] * A["def_gbp_per_l"]
    cost = tot["ann_capex"] + tot["power"] + tot["om"] + tot["transport"] \
        + tot["co2"]
    return dict(mode=mode, n=n_spokes, urea_t_y=urea / 1e3,
                def_t_y=def_kg / 1e3, capex_gbp=tot["capex"],
                cost_gbp_y=cost, revenue_gbp_y=revenue,
                carbon_gbp_y=tot["carbon"], waste_gbp_y=tot["waste"],
                transport_gbp_y=tot["transport"], trips_y=tot["trips"],
                gbp_per_kg_urea=cost / urea,
                net_gbp_y=revenue + tot["carbon"] + tot["waste"] - cost,
                net_no_credits_gbp_y=revenue - cost,
                co2_avoided_t_y=sum(
                    spoke_annual(s)["nh3_y"] for _, s in sel.iterrows())
                / 1e3 * A["co2_avoided_t_per_t_nh3"])


def phased(df, n_final=8, years_per_spoke=2, hub_sizes=(2, 4, 8), horizon=20):
    """NPV of a build-out adding one spoke every `years_per_spoke` years.
    Hub built at year 0 for hub_size spokes' capacity; when the network
    outgrows it, a second train of the same size is added."""
    r = A["discount"]
    sel = df.sort_values("road_km").head(n_final).reset_index(drop=True)
    sp = [spoke_annual(s) for _, s in sel.iterrows()]
    cr = [credits(s, p["nh3_y"]) for (_, s), p in zip(sel.iterrows(), sp)]
    tr = [transport_annual(s)[0] for _, s in sel.iterrows()]
    urea_per = [p["nh3_y"] * MW_UREA / (2 * MW_NH3) / 1e3 for p in sp]
    out = []
    for hs in hub_sizes:
        cap_train = sum(urea_per[:hs]) * 1.05
        for credit_case in ("no credits", "with carbon + waste"):
            npv, trains, rows = 0.0, 0, []
            for y in range(horizon):
                n_on = min(n_final, 1 + y // years_per_spoke)
                cash = 0.0
                need = sum(urea_per[:n_on])
                while trains * cap_train < need:      # build a train
                    cash -= def_plant_capex(cap_train)
                    trains += 1
                if y == 0 or (y % years_per_spoke == 0 and n_on > (1 + (y - 1)
                                                                    // years_per_spoke)):
                    k = n_on - 1                       # new spoke capex
                    cash -= sp[k]["capex"]
                for k in range(n_on):
                    cash -= sp[k]["power"] + sp[k]["om"] + tr[k]
                    if y > 0 and y % int(A["stack_life_y"]) == 0:
                        cash -= A["area_m2_per_kgph_nh3"] * sel.nh3_kg_h[k] \
                            * A["cell_oversize"] * A["stack_gbp_per_m2"]
                urea_kg = need * 1e3
                cash -= urea_kg * A["hub_kwh_per_kg_urea"] / 1e3 \
                    * A["power_gbp_per_mwh"] \
                    + urea_kg * MW_CO2 / MW_UREA * A["co2_gbp_per_kg"] \
                    + trains * (A["maint_frac"] * def_plant_capex(cap_train)) \
                    + A["hub_staff_fte"] * A["fte_gbp"]
                cash += urea_kg / 0.325 / A["def_density"] * A["def_gbp_per_l"]
                if credit_case != "no credits":
                    cash += sum(c + w for c, w in cr[:n_on])
                npv += cash / (1 + r) ** y
                rows.append(dict(year=y, spokes=n_on, trains=trains,
                                 utilisation=need / (trains * cap_train),
                                 cash=cash))
            out.append(dict(hub_size_spokes=hs, credits=credit_case,
                            train_capacity_t_urea_y=cap_train, npv_gbp=npv,
                            final_trains=trains, timeline=rows))
    return out


def main():
    load_plant()
    df = spokes()
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "hub_spoke_sites.csv", index=False)
    print(df[["name", "ml_d", "road_km", "nh3_kg_h", "brine_m3_d",
              "brine_pct_of_flow"]].round(2).to_string(index=False))
    rows = []
    for n in range(1, len(df) + 1):
        rows.append(network(df, n, "hub"))
        rows.append(network(df, n, "local"))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "hub_spoke_sweep.csv", index=False)
    piv = res.pivot(index="n", columns="mode", values="gbp_per_kg_urea")
    print("\nGBP/kg urea (cost, before revenue/credits):\n", piv.round(2))
    hub = res[res["mode"] == "hub"].set_index("n")
    print(hub[["urea_t_y", "capex_gbp", "transport_gbp_y", "trips_y",
               "gbp_per_kg_urea", "net_no_credits_gbp_y", "net_gbp_y",
               "co2_avoided_t_y"]].round(1).to_string())
    # sensitivities on the hub-capex exponent and DEF price at n = 8
    sens = []
    for exp in (0.6, 0.7, 0.8):
        for price in (0.25, 0.30, 0.40):
            for cp in (49.41, 100.0):
                A["hub_exp"], A["def_gbp_per_l"], A["carbon_gbp_per_t"] = \
                    exp, price, cp
                h = network(df, 8, "hub")
                sens.append(dict(hub_exp=exp, def_gbp_per_l=price,
                                 carbon_gbp_per_t=cp,
                                 net_gbp_y=h["net_gbp_y"],
                                 net_no_credits_gbp_y=h["net_no_credits_gbp_y"],
                                 gbp_per_kg_urea=h["gbp_per_kg_urea"]))
    A["hub_exp"], A["def_gbp_per_l"], A["carbon_gbp_per_t"] = 0.6, 0.30, 49.41
    pd.DataFrame(sens).to_csv(OUT / "hub_spoke_sensitivity.csv", index=False)
    ph = phased(df)
    ph_tab = pd.DataFrame([{k: v for k, v in p.items() if k != "timeline"}
                           for p in ph])
    ph_tab.to_csv(OUT / "hub_spoke_phased.csv", index=False)
    print("\nphased build-out (8 spokes, one every 2 y, 20 y):\n",
          ph_tab.round(0).to_string(index=False))
    (OUT / "hub_spoke_summary.json").write_text(json.dumps(
        dict(assumptions=A, hub=HUB), indent=2, default=float))
    plot(df, res, ph)


def plot(df, res, ph):
    hub = res[res["mode"] == "hub"].set_index("n")
    loc = res[res["mode"] == "local"].set_index("n")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    a = ax[0]
    a.plot(hub.index, hub.gbp_per_kg_urea, "o-", label="hub (shared DEF plant)")
    a.plot(loc.index, loc.gbp_per_kg_urea, "s--", label="DEF plant at every spoke")
    rev = hub.revenue_gbp_y / (hub.urea_t_y * 1e3)
    a.plot(hub.index, rev, ":", color="k", label="DEF revenue per kg urea")
    a.plot(hub.index, rev + (hub.carbon_gbp_y + hub.waste_gbp_y)
           / (hub.urea_t_y * 1e3), "-.", color="tab:green",
           label="revenue + carbon + waste credits")
    a.set_xlabel("number of spokes (nearest first)")
    a.set_ylabel("GBP per kg urea")
    a.set_title("Cost per kg urea vs network size")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    a = ax[1]
    w = 0.38
    x = hub.index.values
    a.bar(x - w / 2, hub.net_no_credits_gbp_y / 1e6, w, label="net, no credits")
    a.bar(x + w / 2, hub.net_gbp_y / 1e6, w, label="net, with carbon + waste")
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("number of spokes")
    a.set_ylabel("GBP million per year")
    a.set_title("Hub network annual net result")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "hub_spoke_scale.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for p in ph:
        if p["credits"] != "with carbon + waste":
            continue
        t = pd.DataFrame(p["timeline"])
        ax.step(t.year, t.utilisation * 100, where="post",
                label=f"hub train = {p['hub_size_spokes']} spokes "
                      f"(NPV {p['npv_gbp'] / 1e6:+.1f} M)")
    ax.set_xlabel("year")
    ax.set_ylabel("hub utilisation [%]")
    ax.set_title("Phased build-out: one spoke every 2 years, 8 spokes")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "hub_spoke_phasing.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
