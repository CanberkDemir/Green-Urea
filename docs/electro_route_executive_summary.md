# Green DEF from Nitrate Brine: Electrochemical Route — Executive Summary

Nov 27, 2026 · @SS

## Bottom line

Direct electrochemical nitrate reduction (NO3RR) turns the spent regenerant brine of a drinking-water nitrate plant into anhydrous ammonia and then diesel exhaust fluid (DEF). It needs **20.3 kWh per kg urea** (18.8–25.3 across optimistic–pessimistic cell assumptions). That is 1.6 times less than the electrolyser-hydrogen route modelled on the same IDAES basis (31.8 kWh/kg) and nearly 10 times less than the original paper basis (\~190 kWh/kg).

One plant sized for one large UK ion-exchange (IX) works, such as Yorkshire Water's Keldgate WTW (33 Ml/d through IX), makes about **20 kg/h NH3, 305 t/y urea or 940 t/y DEF**. It avoids about **415 t CO2 per year** of grey ammonia.

At today's assumed costs the plant does **not** pay for itself:

- **Cost:** about £6.4 per kg urea, against DEF revenue worth about £0.85 per kg urea.
- **Credits:** carbon (£49/t) adds only £0.07 per kg urea. The waste credit is larger, at about £0.83 per kg urea, from reusing the denitrified brine as regenerant salt.
- **Net:** about **−£1.4M per year**. The implied CO2 abatement cost is roughly £3,800 per tonne before waste credits.

The route becomes self-funding only if all four conditions in section 6 are met: cheaper and longer-lived electrode stacks, cheap curtailed wind power, higher carbon and DEF prices, and high brine reuse. The case for building it now is as a **waste-to-product nitrogen-recovery service** for water utilities, best deployed as a hub-and-spoke network (companion study). It is not a stand-alone commodity plant.

## Process at a glance

&#91;embedded content: electrochemical route · one Keldgate-size site\]

The cells use most of the power. The NH3 tank lets them follow the wind while the DEF plant runs steadily, and the nitrate-free brine can be reused to regenerate the ion-exchange resin.

## Evidence base

Every design number comes from a published source, a patent or our own IDAES model. The table says which, and how firm it is.

| Design element | Value used | Basis | Strength |
| --- | --- | --- | --- |
| Brine source and size | 33 Ml/d IX works, waste < 0.5 % of output, nitrate rising to 63 mg/l | [Keldgate WTW case study](https://waterprojectsonline.com/wp-content/uploads/case_studies/2009/Keldgate-WTW-2009.pdf) (Yorkshire Water) | Published plant data |
| IX waste fraction | \~0.5 % of treated flow | [Lyng Forge WTW](https://waterprojectsonline.com/wp-content/uploads/case_studies/2006/Lyng-Forge-WTW-2006.pdf) (Anglian Water) | Published plant data |
| Our feed | 204 m3/d, 9.8 g/L nitrate, 1.6 M NaCl | Original IPPS / Aspen brine; matches Keldgate's \~1.8 t/d nitrate in \~165–185 m3/d | Consistent with plant data |
| NO3RR selectivity and rate | FE \~100 % at 1–2 A/cm2 in 1 M nitrate; 4.2 A/cm2 NH3 partial current | [Ru/Cu, OSTI](https://www.osti.gov/pages/biblio/2432566-ammonia-electrosynthesis-from-nitrate-using-rutheniumcopper-cocatalyst-system-full-concentration-range-study); [Chen et al.](https://par.nsf.gov/servlets/purl/10408531) | Lab cells, not brine |
| Dilute-feed performance | FE > 90 % at > 100 mA/cm2 with 2,000 ppm nitrate | [AIChE 2024](https://proceedings.aiche.org/conferences/aiche-annual-meeting/2024/proceeding/paper/707g-direct-electrosynthesis-ammonia-nitrate-reduction-reaction-catalyst-design-and-electrolyzer) | Lab cell |
| Scale-up and cost of flow electrolysers | Scaled membrane cells; TEA below $1.15/kg NH3 in some cases | [RSC EES 2025](https://pubs.rsc.org/en/content/articlepdf/2025/ee/d4ee04382k); [Green Chem 2025](https://pubs.rsc.org/en/content/articlelanding/2025/gc/d5gc04166j) | Reviews and TEA |
| Cell model in this work | 4-stage cascade, FE 0.86, 389 m2, 0.08 A/cm2 average | `electro_no3rr.py` (central case) | Our model, literature ranges |
| NH3 recovery | Rigorous vacuum stripper; 38–42 kW with overhead MVR | `recovery_rigorous.py` (IDAES) | Our model |
| Alternative recovery | Gas-stripping membrane, 7.1 kWh/kg N | [Iddya et al. 2020](https://pubs.rsc.org/en/content/articlelanding/2020/en/c9en01303b) | Published |
| Closest analogue | Catalytic hydrogenation of IX brine nitrate to NH3 + membrane distillation | [Huo et al. 2020, Water Research](https://www.sciencedirect.com/science/article/abs/pii/S0043135420302244) | Published, different reduction step |
| Brine reuse after nitrate removal | Nitrate-free brine can be reused as regenerant | [US 5,616,252](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5616252) | Patent statement |
| DEF plant | Reactor-only total-recycle, 150 bar, 190 C, DEF 32.5 wt % | [US 10,759,745 B2 (Stamicarbon)](https://patents.google.com/patent/US10759745B2/en) | Patent concept |
| DEF quality | Biuret ≤ 0.3 wt %, alkalinity < 0.2 wt % as NH3 | ISO 22241 (via the patent) | Standard |
| Displaced emissions | 2.4 t CO2 per t grey NH3 | [IEA Ammonia Technology Roadmap](https://www.iea.org/reports/ammonia-technology-roadmap/executive-summary) | Agency figure |
| Carbon price | £49.41/t (2026), > £100/t in the 2030s | [GOV.UK UK ETS 2026](https://www.gov.uk/government/publications/determinations-of-the-uk-ets-carbon-price/uk-ets-carbon-price-for-use-in-civil-penalties-2026); [Carbon Pulse](https://carbon-pulse.com/479984/) | Official / modelling |
| Power price | CfD AR6 offshore £58.87/MWh (2012 £) ≈ £82/MWh today | [ORE Catapult](https://ore.catapult.org.uk/resource-hub/blog/allocation-round-6-results-and-analysis) | Auction result |
| Wind profile | Hornsea 2, 2024 hourly capacity factor | Original IPPS input | Measured |

The weakest links are marked "lab cell": no paper yet runs NO3RR in a 1.6 M chloride brine. Section 8 sets out how that is closed.

## Scale-up justification

The scale is set by the brine, not chosen: one large IX works removes about 1.8 t/d of nitrate, which becomes about 20 kg/h of NH3. Each part of the plant is justified differently.

**Electrochemical cells: scaled by numbering up, run gently.**

- The cascade needs about 295 kA over **389 m2** of electrode. That is an average of only **0.08 A/cm2**, 10–50 times below the current densities demonstrated in lab cells (1–4 A/cm2).
- The low current density is forced by mass transfer of dilute nitrate, which falls as conversion rises. That is why the design stages the cells.
- Electrolysers scale by repeating identical cells and stacks. 389 m2 is in the range of one industrial chlor-alkali or alkaline-electrolyser bank (roughly 140–150 elements of \~2.7 m2; approximate). Going from 25 cm2 lab cells to this size is a numbering-up step, not a change of physics.
- What does not transfer from the lab is the **chemistry in brine**: faradaic efficiency and catalyst life in 1.6 M chloride, and whether the cation-exchange membrane keeps chloride away from the anode. Those are the scale-up risks, not size.

**NH3 recovery: conventional units.**

- Vacuum steam stripping with vapour recompression and ammonia rectification are standard unit operations.
- The rigorous IDAES column, with a 1–2 mbar per-stage pressure drop, gives 38–42 kW per 9.85 kg/h NH3.
- The only special item is the recompressor, which needs a pressure ratio of 2.0–2.9 on a 1–1.4 m3/s suction. That is two stages or a turbofan, as in MVR evaporators.

**DEF plant: the scale problem.**

- A world-scale urea plant makes about 3,500 t/d; one spoke supports 0.84 t/d urea.
- The reactor-only patent design keeps the high-pressure section to one reactor and two pumps, but at this size the 150 bar equipment is very small and costly per tonne.
- This is why the DEF step belongs in a shared **hub** fed by several spokes. A combined "aqua-ammonia" variant that skips the rectifier was tested in IDAES and is worse (2.97–3.12 vs 2.08 kWh/kg urea), because the extra water cuts reactor conversion from 0.56 to 0.35–0.38.

**Operation on wind.**

- The cells follow the wind with about 1.3 times oversizing.
- An NH3 tank of about 10 days of DEF feed decouples them from the DEF plant, which runs continuously.
- Storing NH3 instead of electricity was the robust result of both planning models (USD and IPPS bases). Battery firming cost 2.8 times more.

## Plant energy and planning results

The cells are four-fifths of the plant's electricity. All numbers are from IDAES flowsheets with plant-wide pinch analysis (dT\_min 10 K) and heat-pump heating.

| kWh per kg urea | H2 route (electrolyser + hydrogenation) | Electrochemical | Electrochemical + MVR + heat pumps (design) |
| --- | --- | --- | --- |
| NH3 made \[kg/h\] | 11.96 | 20.15 | 20.15 |
| Water electrolysis or NO3RR cells | 24.2 | 16.1 | 16.1 |
| H2 compression, brine concentration | 3.9 | – | – |
| Stripper vapour recompression | – | – | 0.9 |
| Heat pump + e-boiler | 2.6 | 2.9 | 2.7 |
| Chiller + cooling water | 1.0 | 2.0 | 0.5 |
| DEF plant shaft work | 0.2 | 0.2 | 0.2 |
| **Total** | **31.8** | **21.2** | **20.3** |

The design case ranges from 18.8 to 25.3 kWh/kg urea across the cell assumptions. A later optimisation of the DEF plant (stripping steam 0.2 kg/kg CO2, N/C 3.5) cut its share from 0.90 to 0.52 kWh/kg urea.

**Wind-only operation (IPPS model, Hornsea 2 profile).** The same planning model as the original IPPS, with its costs and wind data, sizes the plant for the IPPS demand of 125 kg urea/day:

- **Electrochemical route:** about 227 kW of wind and 23 kWh of battery, with a 1.3 t NH3 tank over the full year.
- **H2 route, original IPPS:** 1,855 kW of wind and 3,150 kWh of battery, on the paper's energy basis.
- **H2 route on the IDAES basis** (10-day horizon, retrained reluMIP surrogates): about 330–350 kW of wind and 42–44 kWh/kg urea.

The grid is never used: at the IPPS cost of capital, wind costs about £0.008/kWh against £0.18/kWh from the grid.

## Economics

A single Keldgate-size plant with its own DEF unit costs **£6.36 per kg urea**. DEF sells for about £0.85 per kg urea equivalent (bulk AdBlue \~30p/l). Two items make up over half the cost: electrode stacks and cell electricity.

| Cost item (one site, 305 t/y urea) | £ per kg urea | Basis |
| --- | --- | --- |
| Electrode stacks (496 m2 incl. 1.3x oversize, £4,700/m2, 5-year life) | 1.86 | `electro_no3rr.py` central; life is an assumption |
| Cell electricity (35.8 kWh/kg NH3 at £82/MWh) | 1.67 | IDAES + CfD AR6 offshore price |
| DEF plant capital (£4.1M) | 1.28 | 0.6-power scaling from a world-scale urea benchmark (assumption) |
| DEF plant O&M and staff | 0.72 | 4 %/y maintenance + 1 FTE (assumption) |
| Stripper, rectifier, tank and civils capital | 0.47 | £1.5M at 20 kg/h NH3 (assumption) |
| Spoke O&M | 0.29 | 4 %/y + 0.5 FTE (assumption) |
| DEF electricity and CO2 feed | 0.06 | IDAES; £20/t CO2 (IPPS) |
| **Total** | **6.36** | 7 % discount rate, 20-year life |

Total capital is about **£8.0M** for one site. Four levers decide viability, in order of size:

1. **Stack cost and life.** £1,000/m2 with a 10-year life cuts the cost by £1.6/kg urea. This is the single largest lever and is the target the electrolyser industry is moving toward.
2. **Power price.** Curtailed or behind-the-meter wind at £40/MWh instead of £82 saves £0.9/kg urea.
3. **Hub instead of local DEF plants.** Sharing one DEF plant across eight spokes brings the cost from £7.6 to £6.1/kg urea (companion study).
4. **Prices.** DEF at 40p/l and carbon at £100/t add about £0.4/kg urea of income together.

With all four, an eight-spoke network reaches about £3.1/kg urea and a small positive net result once waste credits are counted. Without credits it still loses about £2.0M/y.

## Carbon and waste offsets

The waste offset is worth about 12 times the carbon offset for one site, and together they cover about 16 % of the loss.

| Offset (one Keldgate-size site) | Physical amount | Value per year | £ per kg urea |
| --- | --- | --- | --- |
| Grey NH3 displaced (2.4 t CO2/t NH3, IEA) | 173 t NH3, 415 t CO2 | £20k at £49/t; £41k at £100/t | 0.07–0.14 |
| Regenerant salt saved (half the denitrified brine reused, 93.5 kg NaCl/m3, £80/t) | \~3,200 t NaCl | £253k | 0.83 |
| Brine tankering avoided (only where tankered; £10–25/m3 utility tariffs) | up to \~34,000 m3 | £0 at Keldgate; £500k+ at a tankered site | 0–1.7 |
| Nitrogen kept out of the sewer or sea | \~150 t N | not priced | – |

**What this means for the claim.**

- **Carbon alone cannot justify the plant.** Without credits the implied abatement cost is about £4,000 per tonne CO2, against a UK carbon price of £49 today and > £100 in the 2030s. The CO2 avoided is small because the plant is small and DEF's own CO2 is released in use.
- **The defensible case is waste-to-product.** The utility already pays to remove the nitrate and to dispose of or replace the brine. The plant turns that liability into a saleable product and makes brine reuse possible: only a nitrate-free brine can be recycled as regenerant.
- **Tankered sites matter most.** Where the brine is tankered, avoided disposal at £15/m3 adds up to about £1.7 per kg urea. These sites should be the first spokes.

The environmental case is clear: about 150 t of nitrogen a year recovered per site instead of discharged, and 415 t CO2 avoided. The economic case closes only with the cost reductions in section 6.

## Risks and validation plan

The first three items must be demonstrated on real IX brine before any investment decision; the rest are design checks. Each has a go/no-go threshold.

| Risk | Why it matters | Test | Go if |
| --- | --- | --- | --- |
| NO3RR in 1.6 M chloride brine | No published data in brine; chloride may poison the cathode or reach the anode | Bench flow cell on Keldgate or Tophill Low brine, 4-stage cascade | FE ≥ 0.8 at ≥ 0.05 A/cm2, NH3 selectivity ≥ 90 % |
| Catalyst and membrane life | Stacks are the largest cost item; the design assumes 5 years | 2,000 h endurance run with chloride crossover monitoring | Decay < 10 % per 1,000 h; Cl- crossover < 1 % |
| Brine reuse as regenerant | The waste credit depends on it | Regenerate resin with denitrified, pH-adjusted brine | Resin capacity within 5 % of fresh brine |
| Vacuum stripper + MVR | Pressure drop and compressor ratio set the 38–42 kW | Vendor design on the rigorous column | Ratio ≤ 3 in two stages |
| DEF quality | Biuret and NH3 limits (ISO 22241) not modelled | Pilot DEF loop with residence-time control | Biuret ≤ 0.3 wt % |
| Costs | DEF plant and spoke capital are scaled estimates | Vendor quotes for a 1–3 t/d DEF unit | Within ±30 % of the model |

Modelling limits that remain:

- The urea reactor conversion is a trend correlation, not kinetics.
- The planning models are linear, with no minimum-load or start-up costs.
- The DEF plant was modelled at Fco2 3–20 kg/h only.

## Sources

- Keldgate WTW, ACWA Nitreat plant, Yorkshire Water: [UK Water Projects 2009](https://waterprojectsonline.com/wp-content/uploads/case_studies/2009/Keldgate-WTW-2009.pdf)
- Tophill Low WTW, 22 Ml/d IX plant, Yorkshire Water: [UK Water Projects 2015](https://waterprojectsonline.com/wp-content/uploads/case_studies/2015/Tophill-Low-WTW-2015.pdf)
- Lyng Forge WTW nitrate removal, Anglian Water: [UK Water Projects 2006](https://waterprojectsonline.com/wp-content/uploads/case_studies/2006/Lyng-Forge-WTW-2006.pdf)
- Ru/Cu NO3RR full concentration range: [OSTI](https://www.osti.gov/pages/biblio/2432566-ammonia-electrosynthesis-from-nitrate-using-rutheniumcopper-cocatalyst-system-full-concentration-range-study)
- Membrane-free ampere-level NO3RR: [Chen et al., NSF PAR](https://par.nsf.gov/servlets/purl/10408531)
- Dilute-nitrate electrolyser: [AIChE 2024](https://proceedings.aiche.org/conferences/aiche-annual-meeting/2024/proceeding/paper/707g-direct-electrosynthesis-ammonia-nitrate-reduction-reaction-catalyst-design-and-electrolyzer)
- NO3RR scale-up and TEA: [Energy Environ. Sci. 2025](https://pubs.rsc.org/en/content/articlepdf/2025/ee/d4ee04382k); [Green Chem. 2025](https://pubs.rsc.org/en/content/articlelanding/2025/gc/d5gc04166j)
- IX brine nitrate to NH3 with membrane distillation: [Huo et al., Water Research 2020](https://www.sciencedirect.com/science/article/abs/pii/S0043135420302244)
- Gas-stripping membrane, 7.1 kWh/kg N: [Iddya et al., Environ. Sci.: Nano 2020](https://pubs.rsc.org/en/content/articlelanding/2020/en/c9en01303b)
- Brine reuse after nitrate removal: [US 5,616,252](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5616252)
- Low-pressure DEF process: [US 10,759,745 B2](https://patents.google.com/patent/US10759745B2/en)
- Grey ammonia emissions: [IEA Ammonia Technology Roadmap](https://www.iea.org/reports/ammonia-technology-roadmap/executive-summary)
- UK ETS 2026 price: [GOV.UK](https://www.gov.uk/government/publications/determinations-of-the-uk-ets-carbon-price/uk-ets-carbon-price-for-use-in-civil-penalties-2026); 2030s outlook: [Carbon Pulse](https://carbon-pulse.com/479984/)
- CfD AR6 offshore strike price: [ORE Catapult](https://ore.catapult.org.uk/resource-hub/blog/allocation-round-6-results-and-analysis); [GOV.UK results](https://www.gov.uk/government/publications/contracts-for-difference-cfd-allocation-round-6-results)
- Haulage cost: [RHA Cost Tables 2026](https://www.rha.uk.net/Portals/0/Membership/Annual%20Cost%20and%20Pay%20Surveys/Cost%20Tables%202026.pdf)
- Tankered waste tariffs: [Southern Water](https://assetmanager.southernwater.co.uk/businesses/waste-services-charges); [United Utilities](https://corporate.unitedutilities.com/Business-services/tankered-industrial-and-domestic-wastes/)
- Bulk AdBlue prices: [Crown Oil](https://www.crownoil.co.uk/news/adblue-prices)
- Model code: `idaes_model/electro_flowsheet.py`, `electro_cases.py`, `recovery_rigorous.py`, `blite_def.py`, `def_lowp_flowsheet.py`; `optimization/ipps_electro.py`, `hub_spoke.py` (branch idaes-conversion)
