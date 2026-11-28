# Green Ammonia from Nitrate Brine: Direct NH3 Sale vs DEF

Nov 28, 2026 · @SS

## Summary

This study asks whether the nitrate-brine spokes should sell their green ammonia directly instead of sending it to a DEF hub. It uses the same spokes, wind power, transport rates and credits as the hub-and-spoke study (`optimization/hub_spoke.py`); the new model is `optimization/nh3_sale.py`.

- **Selling NH3 directly loses less money than making DEF at every network size.** At 8 spokes the net result with credits is −£2.8M/y for NH3 sale against −£3.7M/y for the DEF hub; at one site it is −£0.96M/y against −£1.58M/y.
- **The reason is the hub, not the product.** DEF earns about three times more per kg of NH3 than ammonia does, but the scaled DEF plant costs more than that extra revenue.
- **No hub is needed, so there is no minimum network size.** Each spoke stands alone and can be built when its water utility is ready.
- **The ammonia price barely matters.** Revenue is only 5–6 % of cost. Raising NH3 from £450/t to £650/t improves the 8-spoke result by only £0.12M/y.
- **The waste credit is the main justification, as in the DEF study.** At 8 spokes it is worth £1.74M/y against £0.07M/y for carbon.
- **Break-even needs the same cost conditions as DEF, but it is reached sooner.** With cheaper, longer-lived stacks, £40/MWh power, £100/t carbon, 80 % brine reuse and a £650/t price with a £300/t green premium, the 8-spoke network makes +£1.21M/y. The DEF hub only just breaks even (+£0.08M/y) under the same conditions.

## Case definition

The spoke is unchanged: the electrochemical NO3RR cascade, vacuum stripper with MVR and rectifier make anhydrous NH3 at about 99.5 %, stored in a small pressure tank. Two ways of selling it were modelled:

| Option | What happens | Freight |
| --- | --- | --- |
| Ex-works | The buyer collects at the spoke gate; the price is net of freight | none to the spoke |
| Delivered to buyer | The spoke trucks NH3 to CF Fertilisers at Billingham, which has imported its ammonia since its own ammonia plant closed in 2023 | £2/km + £300/trip, 10 t loads |

The DEF hub case is the base case of the hub-and-spoke study, recomputed with the same inputs.

**Assumptions new to this study:**

- NH3 price £450/t, near the low end of 2025 NW-Europe grey ammonia prices (about $550–650/t); the green premium is zero in the base case.
- The nine generic spokes have no known location, so their road distance to Billingham is taken as their distance to the hub plus half the hub-to-Billingham leg.

## Results by network size

| Spokes | NH3 t/y | Option | Capex £M | Cost £M/y | Revenue £M/y | Net, no credits £M/y | Net, with credits £M/y |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 173 | Ex-works | 3.85 | 1.31 | 0.08 | −1.23 | −0.96 |
| 1 | 173 | DEF hub | 7.98 | 2.11 | 0.26 | −1.85 | −1.58 |
| 2 | 296 | Ex-works | 6.77 | 2.27 | 0.13 | −2.14 | −1.67 |
| 2 | 296 | DEF hub | 12.48 | 3.31 | 0.44 | −2.86 | −2.39 |
| 8 | 594 | Ex-works | 15.21 | 4.89 | 0.27 | −4.62 | −2.81 |
| 8 | 594 | Delivered | 15.21 | 4.95 | 0.27 | −4.68 | −2.87 |
| 8 | 594 | DEF hub | 23.88 | 6.37 | 0.89 | −5.49 | −3.67 |
| 12 | 799 | Ex-works | 20.96 | 6.67 | 0.36 | −6.31 | −3.58 |
| 12 | 799 | DEF hub | 31.31 | 8.44 | 1.19 | −7.24 | −4.51 |

**Where the money goes (ex-works):** the cost of NH3 is £7.6–8.4/kg. Capital, mostly the electrode stacks replaced every 5 years, is 54 % of it; wind power is 35–39 %.

**Freight to Billingham** adds £83–116/t, about 20–25 % of the NH3 price but only 1 % of total cost. Delivering is therefore a commercial choice, not a design driver.

**Cost per kg rises with more spokes** because the later spokes are smaller (6 kg/h against 20 kg/h at Keldgate) and lose economy of scale. But the later spokes are tankered sites with larger waste credits, so with credits the smaller spokes lose the least per spoke.

## Single-spoke economics

Each spoke is a stand-alone project. Its 20-year NPV at 7 % with ex-works sale:

| Spoke | NH3 kg/h | NPV, no credits £M | NPV, with credits £M |
| --- | --- | --- | --- |
| Keldgate WTW | 20.0 | −13.0 | −10.1 |
| Tophill Low WTW | 14.0 | −9.6 | −7.6 |
| Typical tankered IX works | 6.0 | −4.5 | −2.0 |
| Lyng Forge WTW | 5.0 | −3.9 | −1.8 |

The smallest tankered sites are the best first projects: the avoided tankering and salt make up most of their cost. Keldgate, the largest site, is the worst stand-alone project because its brine is piped away, so it earns no tankering credit.

## Sensitivity

Net £M/y with carbon and waste credits:

| Case | NH3, 1 spoke | NH3, 8 spokes | DEF hub, 8 spokes | NH3 break-even £/t, 8 spokes |
| --- | --- | --- | --- | --- |
| Base | −0.96 | −2.81 | −3.67 | 5,172 |
| NH3 £650/t | −0.92 | −2.69 | −3.67 | 5,172 |
| Green premium +£300/t | −0.90 | −2.63 | −3.67 | 5,172 |
| Stack £1,000/m², 10-year life | −0.46 | −1.10 | −1.96 | 2,293 |
| Power £40/MWh | −0.70 | −1.91 | −2.75 | 3,668 |
| Carbon £100/t | −0.94 | −2.73 | −3.60 | 5,051 |
| Brine reuse 80 % | −0.81 | −1.76 | −2.63 | 3,413 |
| All of the above | +0.06 | +1.21 | +0.08 | −1,093 |

The break-even price is the NH3 price at which the network breaks even with credits. In the base case it is about ten times the market price, so no realistic green premium closes the gap. Stack cost and life matter most, then brine reuse, then power price.

## How to frame it for publication

- **Product choice:** sell NH3 directly. DEF only makes sense if the hub plant can be built far more cheaply than the scaled world-scale benchmark suggests, for example by placing it inside an existing urea or AdBlue site.
- **Justification:** like the DEF study, this is a waste-treatment project with a saleable by-product, not an ammonia business. The CO2 saving is about 1,430 t/y at 8 spokes, and the cost per tonne of CO2 avoided is about £1,970 with credits (vs £2,580 for DEF).
- **Build order:** start with the small tankered sites, not the largest one. No central plant must be committed to first.
- **What would change the answer:** a demonstrated stack life of 10 years at about £1,000/m² and a utility valuing avoided tankering at its full tariff.

## Limits

- The NH3 price is a 2025 range from trade press, not a contract.
- Selling to Billingham depends on CF's willingness to take small 10 t lots; a merchant distributor may pay less ex-works.
- Small anhydrous NH3 tanks at water works fall under COMAH and HSE hazardous-substance consent; aqua ammonia (25 %) would avoid the rectifier and ease storage but has a lower value per tonne. This option was not modelled.
- All other limits of the hub-and-spoke study apply: NO3RR in chloride brine is unvalidated, stack life is assumed, and nine of the twelve spokes are placeholders.

## Files

- Model: `optimization/nh3_sale.py` (imports `hub_spoke.py`)
- Results: `optimization/solutions/nh3_sale_sweep.csv`, `nh3_sale_spoke_npv.csv`, `nh3_sale_sensitivity.csv`, `nh3_sale_summary.json`
- Figure: `figures/nh3_sale_vs_def.png`
