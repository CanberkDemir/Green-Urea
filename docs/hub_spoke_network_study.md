# Green DEF from Nitrate Brine: Hub-and-Spoke Network Study

Nov 27, 2026 · @SS

## Bottom line

**Build the DEF hub at Saltend Chemicals Park, Hull.** Start with the two Yorkshire Water ion-exchange (IX) works within 30 km, Keldgate and Tophill Low, then add spokes over time. A shared hub beats a DEF plant at every site **from the second spoke onward**: £6.32 vs £6.55 per kg urea at two spokes, and £6.08 vs £7.61 at eight.

- **Transport is not the constraint.** NH3 is dense, so eight spokes need about 60 small-tanker trips a year, costing about £36k, or £0.03 per kg urea. Even a spoke 300 km away adds under £0.09 per kg urea.
- **Size the hub for the full programme.** An eight-spoke hub built in one train has the best NPV. Building it in two-spoke modules costs about 5 % more in NPV (£1.9M over 20 years) but caps the risk of stranded capacity if later spokes do not join.
- **The network does not pay at today's costs.**
  - Eight spokes make about 1,050 t/y urea (3,200 t/y DEF) and avoid about 1,430 t/y CO2.
  - They lose about £5.5M/y before credits and £3.7M/y after carbon and waste credits.
  - Carbon is worth only about £70k/y at £49/t. The brine-reuse and tankering credits are worth about £1.7M/y and carry the offset case.
- **What closes the gap.** Cheaper, longer-lived electrode stacks, curtailed wind at £40/MWh, higher salt reuse and higher DEF and carbon prices together bring the network to a small positive net result with credits (section 5).

The defensible claim is environmental and waste-economic, not commercial. The network recovers about 500 t/y of nitrogen from utility waste and lets the utilities recycle their regenerant brine.

## Network concept

&#91;embedded content: hub-and-spoke network · Humber and East Anglia\]

Each spoke runs the NO3RR cells, vacuum stripper and rectifier next to an IX works and keeps two truckloads of NH3. The hub holds the only high-pressure DEF plant. One offshore wind PPA supplies every site.

## Candidate brine sites

The UK has a large installed base of nitrate IX plants making the same kind of brine: ACWA reports its Nitreat process in over 38 UK plants of 3–92 Ml/d ([ACWA](https://www.acwa.co.uk/ion-exchange-nitrate-removal)), and Anglian Water planned nitrate reduction at 22 groundwater sites by treatment or blending in one investment period ([Lyng Forge](https://waterprojectsonline.com/wp-content/uploads/case_studies/2006/Lyng-Forge-WTW-2006.pdf)). Three sites are documented well enough to model.

| Site | Operator | IX flow \[Ml/d\] | Nitrate in → out \[mg/l\] | Brine \[m3/d\] | NH3 \[kg/h\] | Road km to Saltend | Waste route |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Keldgate WTW](https://waterprojectsonline.com/wp-content/uploads/case_studies/2009/Keldgate-WTW-2009.pdf), Cottingham | Yorkshire Water | 33 | 60 → 5 | 185 | 19.7 | 16 | not stated |
| [Tophill Low WTW](https://waterprojectsonline.com/wp-content/uploads/case_studies/2015/Tophill-Low-WTW-2015.pdf) | Yorkshire Water | 22 | 64 → 5 | 132 | 14.1 | 28 | piped 12 km to Beverley WwTW |
| [Lyng Forge WTW](https://waterprojectsonline.com/wp-content/uploads/case_studies/2006/Lyng-Forge-WTW-2006.pdf), Norfolk | Anglian Water | 6 | 77 → 2.5 | 46 | 4.9 | 185 | assumed tankered |
| Typical further IX works (×9, modelled) | Anglian / others | 8 | 70 → 3 | 55 | 5.8 | 60–300 | assumed tankered |

Notes:

- **Flows and nitrate** are from the case studies (Keldgate's "in" is the 63 mg/l forecast; its outlet is "fewer than five ppm"). Brine and NH3 are computed: nitrate removed ÷ 9.8 g/L in the brine, and 95 % conversion to NH3. The computed brine is 0.56–0.76 % of the IX flow, consistent with the stated \~0.5 %.
- **Keldgate matches our design basis almost exactly** (204 m3/d, 20 kg/h NH3), so the IDAES plant is effectively a Keldgate spoke.
- **Road distances** are great-circle × 1.3 from town-level coordinates. The nine "typical" works stand for the Anglian AMP4 programme, whose individual sizes and locations are not in the sources read; they are placeholders, not identified sites.
- **Hub site.** Saltend Chemicals Park is an existing chemical site on the Humber, close to the first two spokes and to the Humber, where the Hornsea offshore wind projects connect. It is a candidate, not an agreed location.

## Model and assumptions

The model (`optimization/hub_spoke.py`) costs every spoke and the hub on one annualised basis (7 % discount rate, 20-year life) and adds spokes nearest-first. Each input is either sourced or marked as an assumption and swept.

| Input | Value | Source or status |
| --- | --- | --- |
| Spoke electricity | 35.8 kWh/kg NH3 | IDAES electrochemical plant with MVR and heat pumps |
| Electrode area | 19.3 m2 per kg/h NH3, 1.3x oversized to follow wind | IDAES cascade; wind-following from the planning models |
| Stack cost and life | £4,700/m2, 5 years | Central case in the cell model; life is an assumption |
| Spoke stripper, rectifier, tank and civils | £1.5M at 20 kg/h NH3, 0.6-power scaling | Assumption |
| DEF hub capital | Urea section = 35 % of a USD 1.6–2.8 bn, 3,500 t/d complex, scaled down with exponent 0.6 (0.7, 0.8 swept) | [Credence Research](https://www.credenceresearch.com/plant-projects/urea-manufacturing-plant-project-report); share and exponent are assumptions |
| DEF hub electricity | 0.6 kWh/kg urea | IDAES DEF plant (0.52 optimised, 0.68 original) |
| Staff | 4 FTE at the hub, 0.5 per spoke, £55k each | Assumption |
| Power | £82/MWh from one offshore wind farm by PPA | CfD AR6 £58.87/MWh in 2012 prices ([ORE Catapult](https://ore.catapult.org.uk/resource-hub/blog/allocation-round-6-results-and-analysis)) |
| Truck cost | £2.0/km plus £300 per trip, 10 t NH3 per load | £208k/y over \~136,800 km for a 44 t artic ([RHA](https://www.rha.uk.net/Portals/0/Membership/Annual%20Cost%20and%20Pay%20Surveys/Cost%20Tables%202026.pdf)) + 30 % pressure-tanker premium (assumption) |
| NH3 storage | £1.89/kg, two loads per spoke | Original IPPS cost |
| CO2 for urea | £20/t | Original IPPS cost |
| DEF price | 30p/l (25–40p swept) | Bulk AdBlue trade quotes of 26–46p/l ([Crown Oil](https://www.crownoil.co.uk/news/adblue-prices)) |
| Carbon | 2.4 t CO2/t NH3 displaced, £49.41/t (£100 swept) | [IEA](https://www.iea.org/reports/ammonia-technology-roadmap/executive-summary); [UK ETS 2026](https://www.gov.uk/government/publications/determinations-of-the-uk-ets-carbon-price/uk-ets-carbon-price-for-use-in-civil-penalties-2026) |
| Waste credit | Half of the denitrified brine reused: saves 93.5 kg NaCl/m3 at £80/t, plus £15/m3 tankering where tankered | Reuse: [US 5,616,252](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5616252); tariffs £10–25/m3 ([Southern Water](https://assetmanager.southernwater.co.uk/businesses/waste-services-charges)); share and salt price are assumptions |

**Production schedule.** Every spoke's cells follow the shared wind farm. Its NH3 tank holds two truckloads, so a 20 kg/h spoke ships a 10 t load about every three weeks. The hub keeps its own NH3 inventory and runs continuously. The planning models showed that storing NH3 instead of electricity is the cheaper way to ride through low-wind days.

## Results

The hub's cost per kg urea falls with every spoke, while separate DEF plants get worse as smaller sites join. Transport stays below 1 % of cost.

| Spokes | Urea \[t/y\] | Capital \[£M\] | Hub cost \[£/kg urea\] | DEF at every spoke \[£/kg urea\] | Truck trips \[/y\] | Net, no credits \[£M/y\] | Net, with credits \[£M/y\] | CO2 avoided \[t/y\] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 305 | 8.0 | 6.92 | 6.36 | 17 | −1.85 | −1.58 | 415 |
| 2 | 523 | 12.5 | 6.32 | 6.55 | 30 | −2.86 | −2.39 | 712 |
| 4 | 703 | 16.5 | 6.21 | 7.08 | 40 | −3.77 | −2.84 | 957 |
| 8 | 1,048 | 23.9 | 6.08 | 7.61 | 59 | −5.49 | −3.67 | 1,426 |
| 12 | 1,408 | 31.3 | 5.99 | 7.87 | 80 | −7.24 | −4.51 | 1,917 |

The hub wins from two spokes because its fixed staff and its high-pressure equipment are shared. With one spoke, the haul and the hub's larger staff make it dearer than a local plant.

Costs dominate the result whatever the network size. At eight spokes, cost per kg urea breaks down as follows:

- electrode stacks £1.86 and cell electricity £1.67;
- hub capital £0.78 and hub O&M £0.54;
- spoke equipment £0.65 and spoke O&M £0.48;
- transport £0.03.

**What it takes to break even (eight spokes).**

| Step (cumulative) | Cost \[£/kg urea\] | Net with credits \[£M/y\] | Net, no credits \[£M/y\] |
| --- | --- | --- | --- |
| Base case | 6.08 | −3.67 | −5.49 |
| Stack £1,000/m2, 10-year life | 4.45 | −1.96 | −3.78 |
| + power £40/MWh (curtailed wind, behind the meter) | 3.57 | −1.04 | −2.86 |
| + carbon £100/t, DEF 40p/l | 3.57 | −0.67 | −2.56 |
| + salt reuse 80 %, spoke equipment £0.75M, no cell oversizing | 3.08 | +0.89 | −2.04 |

Sensitivity: a hub capital exponent of 0.7 or 0.8 instead of 0.6 cuts the eight-spoke cost to £5.52 or £5.24 per kg urea. DEF at 40p/l instead of 30p adds about £0.3M/y.

## Phased build-out and hub size

Add one spoke every two years, nearest first, until there are eight. Build the hub for the whole programme if the later spokes are committed, otherwise in two-spoke modules. The difference is about 5 % of NPV.

| Hub built as | Capacity per train \[t urea/y\] | Trains by year 20 | NPV, no credits \[£M\] | NPV, with credits \[£M\] | Utilisation |
| --- | --- | --- | --- | --- | --- |
| One train for 8 spokes | 1,100 | 1 | −46.8 | **−35.6** | 28 % in year 1, 95 % from year 14 |
| Trains for 2 spokes each | 549 | 2 | −48.7 | −37.5 | 56 % in year 1, 95 % in years 2–3, 56 % again when train 2 starts, 95 % from year 14 |
| Trains for 4 spokes each | 738 | 2 | −49.6 | −38.3 | 41 % in year 1, then 54–95 % |

- **Why one big train wins.** Under 0.6-power scaling, a plant twice the size costs only about 1.5 times as much. That outweighs the cost of capacity sitting idle in the early years.
- **Why modules are still a sensible choice.** The single train only pays off if all eight spokes join. Here the first two spokes, Keldgate and Tophill Low, already fill a 2-spoke module to 95 %. Each later spoke depends on another utility's decision. A second module can be added when spokes 3–6 are contracted, for an NPV penalty of about £1.9M.
- **Recommended sequence.**
  1. Years 0–2: Keldgate spoke and a 2-spoke hub module at Saltend; demonstrate brine reuse.
  2. Year 2: Tophill Low, filling the first module.
  3. Years 4–8: tankered Anglian sites, chosen by brine volume, not distance. Add the second module once two of them are signed.
  4. Beyond: spokes up to 300 km away stay viable, since haulage adds under £0.1 per kg urea.

The phasing figure (utilisation by year) is in `figures/hub_spoke_phasing.png` and the cost-vs-spokes figure in `figures/hub_spoke_scale.png`.

## Carbon and waste offsets

At eight spokes, offsets recover about £1.8M/y of the £5.5M/y loss. Waste credits are 25 times larger than carbon credits.

| Offset, eight spokes | Physical amount per year | Value per year |
| --- | --- | --- |
| Grey NH3 displaced | \~590 t NH3, \~1,430 t CO2 | £70k at £49/t; £143k at £100/t |
| Regenerant salt saved and tankering avoided (half the brine reused) | \~116,000 m3 of brine reused | \~£1.74M |
| Nitrogen recovered instead of discharged | \~510 t N | not priced |
| Nitrate removed from the brine stream | \~2,300 t NO3 | not priced |

Three points for the argument:

1. **Carbon does not justify the network on price.** The implied abatement cost is about £3,850 per tonne CO2 before waste credits, against £49 today and > £100 in the 2030s.
2. **Waste offsets are the real case.** Utilities already pay for regenerant salt and, at many sites, for tankering. Only nitrate-free brine can be reused, so the network turns a cost the utilities carry anyway into a recycled input and a product.
3. **The value depends on site selection.** A tankered site earns about three times the credit of a site that drains to sewer. Choosing tankered sites first is the cheapest way to improve the case.

Not counted, and worth adding in a full life-cycle study:

- the embodied carbon of stacks and plant;
- the wind farm's share of emissions;
- avoided brine haulage emissions;
- the nitrogen load removed from receiving waters.

## Risks and open questions

The network inherits the electrochemical route's technical risks and adds commercial ones. The biggest unknowns are site data and utility participation.

| Risk or question | Effect | How to close it |
| --- | --- | --- |
| Brine composition and volume per site | Spoke size and NH3 output | Utility data requests to Yorkshire and Anglian Water; brine sampling at Keldgate and Tophill Low |
| Whether each site tankers its brine | Waste credit varies about threefold | Utility disposal records; Anglian Water's 2025 brine tender |
| Brine reuse as regenerant after NO3RR | Most of the offset value | Resin regeneration trial with denitrified brine |
| NO3RR in chloride brine, stack life | Largest cost items | Bench and endurance tests (executive summary, section 8) |
| DEF hub capital | Scaled from a world-scale benchmark | Vendor quotes for 1–3 t/d DEF units; check against modular units |
| NH3 haulage permitting (ADR, COMAH) | Hub and spoke site licensing | Early engagement with HSE and the Environment Agency; small-quantity storage limits |
| Utilities' willingness to host spokes | Phasing and hub utilisation | Start with one operator (Yorkshire Water) and modular hub trains |
| Hub site availability | Location | Saltend and Immingham options |

Modelling limits that remain:

- Spoke and hub costs are annualised, not scheduled hour by hour. The hourly wind dispatch was modelled separately for one plant.
- Nine of the twelve spokes are generic placeholders.
- Road distances are estimates.

## Sources

- Keldgate WTW: [UK Water Projects 2009](https://waterprojectsonline.com/wp-content/uploads/case_studies/2009/Keldgate-WTW-2009.pdf)
- Tophill Low WTW: [UK Water Projects 2015](https://waterprojectsonline.com/wp-content/uploads/case_studies/2015/Tophill-Low-WTW-2015.pdf)
- Lyng Forge WTW and the Anglian AMP4 programme: [UK Water Projects 2006](https://waterprojectsonline.com/wp-content/uploads/case_studies/2006/Lyng-Forge-WTW-2006.pdf)
- Nitreat installed base: [ACWA](https://www.acwa.co.uk/ion-exchange-nitrate-removal)
- Brine reuse after nitrate removal: [US 5,616,252](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5616252)
- Urea complex capital: [Credence Research](https://www.credenceresearch.com/plant-projects/urea-manufacturing-plant-project-report)
- Haulage cost: [RHA Cost Tables 2026](https://www.rha.uk.net/Portals/0/Membership/Annual%20Cost%20and%20Pay%20Surveys/Cost%20Tables%202026.pdf)
- Offshore wind price: [ORE Catapult on CfD AR6](https://ore.catapult.org.uk/resource-hub/blog/allocation-round-6-results-and-analysis); [GOV.UK AR6 results](https://www.gov.uk/government/publications/contracts-for-difference-cfd-allocation-round-6-results)
- Grey ammonia emissions: [IEA Ammonia Technology Roadmap](https://www.iea.org/reports/ammonia-technology-roadmap/executive-summary)
- UK carbon price: [GOV.UK UK ETS 2026](https://www.gov.uk/government/publications/determinations-of-the-uk-ets-carbon-price/uk-ets-carbon-price-for-use-in-civil-penalties-2026); [Carbon Pulse, 2030s outlook](https://carbon-pulse.com/479984/)
- Tankered waste tariffs: [Southern Water](https://assetmanager.southernwater.co.uk/businesses/waste-services-charges); [United Utilities](https://corporate.unitedutilities.com/Business-services/tankered-industrial-and-domestic-wastes/)
- Bulk AdBlue price: [Crown Oil](https://www.crownoil.co.uk/news/adblue-prices)
- Companion document: Green DEF from Nitrate Brine: Electrochemical Route — Executive Summary
- Model: `optimization/hub_spoke.py`; results in `optimization/solutions/hub_spoke_*.csv` (branch idaes-conversion)
