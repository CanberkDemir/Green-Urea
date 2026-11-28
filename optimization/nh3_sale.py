"""
Direct sale of green NH3 from the nitrate-brine spokes (no DEF hub).

Same spokes, power, credits and costs as hub_spoke.py; the product is
anhydrous NH3 sold to a bulk buyer instead of being converted to DEF.
Delivery options:
  ex-works - buyer collects at the spoke gate (price net of freight);
  buyer    - spoke trucks NH3 to CF Fertilisers Billingham (UK NH3 consumer
             that imports NH3 since its ammonia plant closed in 2023) at the
             hub_spoke ADR tanker rates.
Compared with the DEF hub at the same number of spokes.

    python nh3_sale.py
Writes optimization/solutions/nh3_sale_*.csv/json and figures/nh3_sale_vs_def.png
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import hub_spoke as H  # noqa: E402

A = H.A
N = dict(
    # NW Europe grey NH3 ~ USD 550-650/t in 2025 (trade press ranges);
    # base GBP 450/t; green premium as a sensitivity (assumption)
    nh3_gbp_per_t=450.0,
    green_premium_gbp_per_t=0.0,
    buyer=dict(name="CF Fertilisers, Billingham", lat=54.596, lon=-1.284),
)


def sites():
    df = H.spokes()
    n_gen = len(H.GENERIC_KM)
    lat = [s[5] for s in H.SITES] + [None] * n_gen
    lon = [s[6] for s in H.SITES] + [None] * n_gen
    hb = A["road_factor"] * H.gc_km((H.HUB["lat"], H.HUB["lon"]),
                                    (N["buyer"]["lat"], N["buyer"]["lon"]))
    # generic spokes: buyer distance = hub distance + 0.5 x hub-buyer leg
    # (assumption; their locations are unknown)
    df["buyer_km"] = [A["road_factor"] * H.gc_km((la, lo), (N["buyer"]["lat"],
                                                          N["buyer"]["lon"]))
                      if la is not None else rk + 0.5 * hb
                      for la, lo, rk in zip(lat, lon, df.road_km)]
    return df


def spoke(s, delivery):
    sp = H.spoke_annual(s)
    carbon, waste = H.credits(s, sp["nh3_y"])
    freight = trips = 0.0
    if delivery == "buyer":
        trips = sp["nh3_y"] / (A["truck_payload_t"] * 1e3)
        freight = trips * (2 * s.buyer_km * A["truck_gbp_per_km"]
                           + A["truck_fixed_gbp"])
    price = N["nh3_gbp_per_t"] + N["green_premium_gbp_per_t"]
    cost = sp["ann_capex"] + sp["power"] + sp["om"] + freight
    return dict(cost=cost, rev=sp["nh3_y"] / 1e3 * price, carbon=carbon,
                waste=waste, freight=freight, trips=trips, nh3_y=sp["nh3_y"],
                capex=sp["capex"], ann_capex=sp["ann_capex"],
                power=sp["power"], om=sp["om"])


def network(df, n, delivery):
    sel = df.sort_values("road_km").head(n)
    t = None
    for _, s in sel.iterrows():
        r = spoke(s, delivery)
        t = r if t is None else {k: t[k] + r[k] for k in t}
    nh3 = t["nh3_y"]
    return dict(delivery=delivery, n=n, nh3_t_y=nh3 / 1e3,
                capex_gbp=t["capex"], cost_gbp_y=t["cost"],
                revenue_gbp_y=t["rev"], carbon_gbp_y=t["carbon"],
                waste_gbp_y=t["waste"], freight_gbp_y=t["freight"],
                trips_y=t["trips"], gbp_per_kg_nh3=t["cost"] / nh3,
                capex_share=t["ann_capex"] / t["cost"],
                power_share=t["power"] / t["cost"],
                net_no_credits_gbp_y=t["rev"] - t["cost"],
                net_gbp_y=t["rev"] + t["carbon"] + t["waste"] - t["cost"],
                co2_avoided_t_y=nh3 / 1e3 * A["co2_avoided_t_per_t_nh3"],
                breakeven_nh3_gbp_per_t=(t["cost"] - t["carbon"] - t["waste"])
                / (nh3 / 1e3))


def spoke_npv(s, with_credits):
    """20-y NPV of one stand-alone spoke selling NH3 ex-works."""
    r = A["discount"]
    sp = H.spoke_annual(s)
    c, w = H.credits(s, sp["nh3_y"])
    rev = sp["nh3_y"] / 1e3 * (N["nh3_gbp_per_t"] + N["green_premium_gbp_per_t"])
    stack = A["area_m2_per_kgph_nh3"] * s.nh3_kg_h * A["cell_oversize"] \
        * A["stack_gbp_per_m2"]
    npv = -sp["capex"]
    for y in range(1, A["life_y"] + 1):
        cash = rev - sp["power"] - sp["om"] + ((c + w) if with_credits else 0)
        if y % int(A["stack_life_y"]) == 0 and y < A["life_y"]:
            cash -= stack
        npv += cash / (1 + r) ** y
    return npv


def main():
    H.load_plant()
    df = sites()
    rows = []
    for n in range(1, len(df) + 1):
        rows += [network(df, n, "ex-works"), network(df, n, "buyer")]
        h = H.network(df, n, "hub")
        rows.append(dict(delivery="DEF hub", n=n,
                         nh3_t_y=h["urea_t_y"] * 2 * H.MW_NH3 / H.MW_UREA,
                         capex_gbp=h["capex_gbp"], cost_gbp_y=h["cost_gbp_y"],
                         revenue_gbp_y=h["revenue_gbp_y"],
                         carbon_gbp_y=h["carbon_gbp_y"],
                         waste_gbp_y=h["waste_gbp_y"],
                         net_no_credits_gbp_y=h["net_no_credits_gbp_y"],
                         net_gbp_y=h["net_gbp_y"],
                         co2_avoided_t_y=h["co2_avoided_t_y"]))
    res = pd.DataFrame(rows)
    res.to_csv(H.OUT / "nh3_sale_sweep.csv", index=False)
    cols = ["n", "delivery", "nh3_t_y", "capex_gbp", "cost_gbp_y",
            "revenue_gbp_y", "carbon_gbp_y", "waste_gbp_y",
            "net_no_credits_gbp_y", "net_gbp_y"]
    print(res[res.n.isin([1, 2, 3, 8, 12])][cols].round(0).to_string(index=False))
    ex = res[res.delivery == "ex-works"].set_index("n")
    print("\nex-works:\n", ex[["gbp_per_kg_nh3", "capex_share", "power_share",
                               "breakeven_nh3_gbp_per_t"]].round(3).to_string())
    by = res[res.delivery == "buyer"].set_index("n")
    print("\nfreight to buyer GBP/t:\n",
          (by.freight_gbp_y / by.nh3_t_y).round(1).to_string())
    print("\nbuyer km:\n", df[["name", "road_km", "buyer_km"]].round(0).to_string())

    npv = [dict(name=s["name"], nh3_kg_h=s.nh3_kg_h,
                npv_no_credits=spoke_npv(s, False),
                npv_with_credits=spoke_npv(s, True))
           for _, s in df.sort_values("road_km").iterrows()]
    pd.DataFrame(npv).to_csv(H.OUT / "nh3_sale_spoke_npv.csv", index=False)
    print("\nspoke NPV (ex-works, 20 y):\n",
          pd.DataFrame(npv).round(0).to_string(index=False))

    base = dict(A)
    cases = {
        "base": {},
        "NH3 GBP 650/t": dict(nh3=650.0),
        "green premium +GBP 300/t": dict(prem=300.0),
        "stack GBP 1000/m2, 10 y": dict(stack_gbp_per_m2=1000.0, stack_life_y=10.0),
        "power GBP 40/MWh": dict(power_gbp_per_mwh=40.0),
        "carbon GBP 100/t": dict(carbon_gbp_per_t=100.0),
        "salt reuse 80 %": dict(salt_reuse_frac=0.8),
        "all of the above": dict(nh3=650.0, prem=300.0, stack_gbp_per_m2=1000.0,
                                 stack_life_y=10.0, power_gbp_per_mwh=40.0,
                                 carbon_gbp_per_t=100.0, salt_reuse_frac=0.8),
    }
    sens = []
    for nm, c in cases.items():
        A.update(base)
        N["nh3_gbp_per_t"] = c.get("nh3", 450.0)
        N["green_premium_gbp_per_t"] = c.get("prem", 0.0)
        A.update({k: v for k, v in c.items() if k not in ("nh3", "prem")})
        for n in (1, 8):
            e = network(df, n, "ex-works")
            h = H.network(df, n, "hub")
            sens.append(dict(case=nm, n=n, nh3_net=e["net_gbp_y"],
                             nh3_net_no_credits=e["net_no_credits_gbp_y"],
                             def_net=h["net_gbp_y"],
                             breakeven_nh3=e["breakeven_nh3_gbp_per_t"]))
    A.update(base)
    N["nh3_gbp_per_t"], N["green_premium_gbp_per_t"] = 450.0, 0.0
    sdf = pd.DataFrame(sens)
    sdf.to_csv(H.OUT / "nh3_sale_sensitivity.csv", index=False)
    print("\nsensitivity (GBP/y, with credits):\n",
          sdf.round(0).to_string(index=False))
    (H.OUT / "nh3_sale_summary.json").write_text(json.dumps(
        dict(nh3=N, assumptions=A), indent=2, default=float))
    plot(res)


def plot(res):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for dlv, st in (("ex-works", "o-"), ("buyer", "s--"), ("DEF hub", "^:")):
        r = res[res.delivery == dlv].set_index("n")
        ax[0].plot(r.index, r.net_no_credits_gbp_y / 1e6, st, label=dlv)
        ax[1].plot(r.index, r.net_gbp_y / 1e6, st, label=dlv)
    for a, t in zip(ax, ("Net without credits", "Net with carbon + waste credits")):
        a.axhline(0, color="k", lw=0.8)
        a.set_xlabel("number of spokes (nearest first)")
        a.set_ylabel("GBP million per year")
        a.set_title(t)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("Selling green NH3 directly vs converting to DEF at a hub")
    fig.tight_layout()
    fig.savefig(H.FIG / "nh3_sale_vs_def.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
