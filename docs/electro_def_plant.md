# Electrochemical NH3 + low-pressure DEF plant: design, energy, flexible operation

Session log, 2026-11-26. Everything here is reproducible:

| Step | Command (from `idaes_model/` unless noted) |
|---|---|
| Electro section alone | `python electro_flowsheet.py [--mvr] [--case optimistic]` |
| Case comparison, charts, block PFDs, JSON | `python electro_cases.py` (~5 min) |
| Unit-level PFD | `python draw_pfd.py electro` |
| Whole-plant table incl. old cases | `python plant_energy.py [--electro-only]` |
| Flexible operation LP | `cd optimization; python flex_planning.py` (~2 min) |

## 1. Solver fix (TODO D6)
The DEF flowsheet does not converge with the IDAES Windows Ipopt 3.13.2.
With the official COIN-OR Ipopt **3.14.20** build
(`C:\Users\Sunoy\ipopt-3.14`, or set `IPOPT_314_EXE`), it solves in about
25 s. `solver_path.ipopt314()` switches Pyomo to that build only around the
DEF and electro solves. The ammonia flowsheet's B6 initialization fails
under 3.14, so it stays on 3.13.2.

Still open: the condensate-to-DEF option (`--direct > 0`) is infeasible even
on 3.14. That holds when the split is ramped up from the converged all-recycle
case, so it is a model/spec problem, not the solver. `plant_energy.py` now
uses `direct=0` (all condensate recycled, the patent's base case).

## 2. Which urea route after electrochemical NH3?
At this scale the plant makes 20 kg/h NH3, which becomes 36 kg/h urea or
110 kg/h DEF (~2.6 t/day).

| Option | Verdict |
|---|---|
| Direct CO2 + NO3- co-electrolysis to urea | Rejected: FE tens of %, ~0.3 wt% urea in 1.6 M NaCl, cannot be separated cheaply (see `electro_no3rr_survey.md`) |
| Conventional CO2-stripping urea plant (Stamicarbon/Saipem type) + evaporation + granulation | Rejected at 36 kg/h: HP stripper, scrubber and evaporation/granulation train are designed for >1000 t/d; solid urea also needs ~99.7 % concentration |
| Once-through reactor, unconverted NH3 sold as ammonium salt | Rejected: wastes the expensive electro-NH3 |
| Aqua-ammonia feed (skip the rectifier) | Rejected: ~5 mol% NH3 overhead puts H2O/CO2 >> 1 in the reactor and collapses equilibrium conversion |
| **Low-pressure reactor-only total-recycle plant making DEF (US 10,759,745 concept, `def_lowp_flowsheet.py`)** | **Chosen**. DEF (32.5 wt% urea) needs no evaporation or prilling, sells as a finished product, and the 150 bar reactor-only loop has the fewest HP units |

So the patent route stays. The improvements are around it:
1. **NH3 buffer tank between the sections.** Electro-NH3 follows the wind and
   the DEF plant runs continuously or slowly (section 5).
2. **Plant-wide heat integration.** The 65 °C cell heat and the DEF
   condensers (90-120 °C) cover the vacuum stripper and brine preheat.
3. **Stripper-overhead MVR.** Recompressing the overhead from 0.08 to 0.25 bar
   lets it condense at ~60 °C instead of being chilled to 10 °C.
4. **Heating by temperature.** Heat pumps cover heat up to 160 °C (sourced
   below the pinch where possible). Only the >160 °C duties use an e-boiler:
   the rectifier reboiler at 198 °C and the 190 °C reactor.

## 3. IDAES model of the electrochemical section (`electro_flowsheet.py`)
Feed, then preheat to 65 °C, then the cell (StoichiometricReactor; extents
and power come from the `electro_no3rr.py` 4-stage cascade, central case),
then an H2 degassing flash, then a **rigorous 10-stage vacuum stripper**
(0.08 bar, 99 % recovery, `nh3_stripper.build_column`), then a shortcut
15 bar rectifier fed with the rigorous overhead.

Rigorous result:
- reboiler 299 kW at 42.5 °C;
- overhead only 4.9 mol% NH3, because the raw brine is dilute;
- 0.16 kg/h NH3 is lost with the HER hydrogen.

PFD: `figures/electro_pfd.png`; block PFDs: `figures/block_pfd_*.png`.

## 4. Energy (plant-wide heat integration, dT_min 10 K)
`figures/electro_energy_by_category.png`, `figures/electro_unit_duties.png`,
`figures/electro_gcc_*.png`, numbers in `figures/electro_cases.json`.

| kWh per kg urea | classical (case 2) | electro | **electro+** (MVR + pinch heat pump) |
|---|---|---|---|
| NH3 made [kg/h] | 11.96 | 20.15 | 20.15 |
| water electrolysis / NO3RR cells | 24.2 | 16.1 | 16.1 |
| H2 compression, B6 | 3.9 | – | – |
| stripper MVR | – | – | 0.9 |
| heat pump + e-boiler | 2.6 | 2.9 | 2.7 |
| chiller + cooling water | 1.0 | 2.0 | 0.5 |
| DEF shaft | 0.2 | 0.2 | 0.2 |
| **total** | **31.8** | **21.2** | **20.3** (18.8 opt., 25.3 pess.) |

(Paper: ~190 kWh/kg urea.)

The DEF/urea section is small: about 0.2 kWh/kg shaft plus its share of the
heat. On its own (`figures/electro_gcc_DEF.png`) its condensers can supply
the LP decomposer and stripping steam, and only the 190 °C reactor and the
160 °C MP decomposer need lifted heat. The urea side therefore needs no
dedicated integration network beyond the plant-wide one. The cells (16 kWh/kg
urea) and the >160 °C e-boiler duty are what matter.

### Bug fixed on the way
`heat_integration.PinchResult.cold_below` and `hot_above` read the grand
composite curve on the wrong side: they returned the heat released *above* T
instead of below it. They have been checked against hand cases. This changes
the chilled-water split in `ammonia_energy.py` and `plant_energy.py`
slightly.

## 5. Flexible operation on wind only (`optimization/flex_planning.py`)
An hourly LP over 2024 (Hornsea-2 profile) sizes the wind, battery, NH3 tank
and NO3RR capacity. Annualised costs:

| Item | Cost basis |
|---|---|
| Wind | $1500/kW, 25 y, 8 %, plus O&M |
| Battery | $250/kWh, 15 y |
| Tank | $3/kg NH3, 25 y |
| Stack | $6000/m² electrode, 10 y |

| Strategy | Wind kW | Battery kWh | NH3 tank | NO3RR capacity | $/kg urea |
|---|---|---|---|---|---|
| everything continuous, battery-firmed | 3880 | 29 200 | – | 1.0x | **6.66** |
| NH3 follows wind + NH3 tank, DEF continuous | 1480 | 157 | 10 t (~500 h) | 1.34x | **2.41** |
| ... + DEF flexible 60-120 % | 1475 | 73 | 1.1 t (~55 h) | 1.34x | **2.38** |

`figures/flex_planning_costs.png`, `figures/flex_planning_dispatch.png`.

Conclusions:
- **Store NH3, not electricity.** Firming with batteries costs 2.8x more.
- **Oversize the cells by about 1.3x** and let them ramp. They seldom stop
  completely; the LP mostly modulates them.
- **Keep the DEF plant running.** Letting it flex by ±20-40 % saves little
  (1 %) but shrinks the NH3 tank 9x. That makes a turndown-capable DEF loop
  worthwhile mostly for safety and inventory: anhydrous NH3 stock drops from
  10 t to 1 t.
- **The stack dominates the remaining cost** (~$467k/yr of ~$740k/yr), so
  electrode $/m² and lifetime are the main levers. Electricity is second.

Limits: this is an LP, so there are no minimum-load or start-up penalties.
The stripper and rectifier turndown (~30 %) and the hot-standby power of a
stopped DEF loop are not modelled. The heat-integration credit is assumed to
scale with the NH3 rate. No grid, no DEF or brine storage costs, no H2
by-product credit.

## 6. Open items
- D6b: condensate-to-DEF (`--direct`) infeasible at fixed X.
- The rectifier reboiler at 198 °C forces an e-boiler. Trying 6 bar (160 °C
  sump, NH3 condensing at 10 °C) would let a heat pump serve it.
- The NO3RR parameters are literature ranges. The chloride-brine cell, CEM
  H+/Na+ selectivity and catalyst life are undemonstrated.
- Add minimum-load and start-up constraints (MILP) to the planning model, and
  costs for the DEF plant and the classical route for a like-for-like
  comparison.

## 7. Results on the existing IPPS basis (`optimization/ipps_electro.py`)
This model reuses the IPPS model's own inputs by importing them from
`ipps_green_urea_fixed_operating_point_relu_omlt.py`:
- wind file and horizon sampling (888 h, one random day per 10-day block);
- costs: GBP, CRF 0.05, wind £864.5/kW, battery £104/kWh, NH3 storage
  £1.89/kg, grid £0.18/kWh, CO2 £0.02/kg, O2 credit £0.03/kg;
- 1C battery, demand 125 kg urea/day, urea always on, the four grid modes.

The electro plant's electricity per kg comes from `electro_cases.json`
(electro+ central). The stack is costed at £4700/m². It is an LP; the cells
modulate.

| wind_only | H2 route (existing IPPS results) | electro, IPPS horizon | electro, full year |
|---|---|---|---|
| Wind [kW] | 1855 | 218 | 227 |
| Battery [kWh] | 3150 | 153 | 23 |
| NH3 storage [kg] | 483 | 134 | 1342 (~10 days of DEF feed) |
| Electricity [kWh/kg urea] | 193 | 20.4 | 20.4 |
| £/kg urea excl. plant-capex intercept | 4.12 | 0.53 | 0.55 |
| £/kg urea incl. intercept | 319 | 315.6 | 315.6 |

`figures/ipps_h2_vs_electro.png`; CSVs in `optimization/solutions/ipps_electro_*.csv`.

How to read this:
1. **Energy basis.** The H2-route numbers use the IPPS surrogates trained
   on the Aspen grids (paper basis, ~190 kWh/kg urea). The IDAES model of
   that same route gives 31.8 kWh/kg (section 4). Most of the 8-9x gap in
   wind and cost is that energy basis, not the route. Like-for-like
   (IDAES vs IDAES), the electro route uses 1.6x less electricity and needs
   no electrolyser or H2 storage. To compare properly, retrain the IPPS
   surrogates on IDAES data (TODO F4) and rerun both routes.
2. **The plant-capex correlation has intercepts** of £191.6M (NH3) and
   £95.8M (urea). At 46 t urea/yr they set ~99 % of the annualised cost
   (£14.4M/yr) whatever the design, so they hide every other trade-off.
   They are probably meant for a much larger plant; this should be checked
   with whoever set them.
3. **Grid never pays here.** With CRF 0.05 and no O&M, wind costs about
   £0.008/kWh against £0.18/kWh grid, so all four grid modes give the same
   design.
4. **Sampled vs full year.** The sampled horizon stitches non-consecutive
   days into one cycle, which undervalues multi-day storage: it chooses a
   small tank plus a 153 kWh battery. The full year chooses a 1.3 t NH3
   tank and almost no battery. **Storing NH3, not electricity, is the
   robust conclusion**, the same as the USD model in section 5.
5. **G2 result.** A 6.2 bar rectifier gives 20.7 vs 20.3 kWh/kg urea, so it
   does not help. Its 160 °C sump still needs a hot utility above the
   160 °C heat-pump limit, and the 190 °C DEF reactor stays on the e-boiler.

## 8. IPPS surrogates retrained on IDAES data with reluMIP (TODO F4, in progress)
Session 2026-11-26/27. Stopped by the user mid-iteration; state below.

### Files
| File | Role |
|---|---|
| `surrogates/idaes_data.py` | IDAES sampling per section (`.venv`): `A`, `B`, `C`, `D`; `--ft/--fh2/--fco2/--step` add points to the existing CSVs (domain expansion) |
| `surrogates/idaes_data/{A,B,C,D}.csv` | training data |
| `surrogates/idaes_train_relumip.py` | one Keras ReLU net (8,8) per output, scaling folded in; every net is embedded with reluMIP (big-M, LP bound tightening, Gurobi) and checked against Keras (system Python) |
| `surrogates/relumip_compat.py` | reluMIP fix: its node-redundancy check indexes past the hidden layers whenever an output never changes sign (every flow/power here) -> IndexError |
| `surrogates/idaes_surrogate_plots.py [oppoint.json]` | data + fitted surfaces, IPPS operating points marked; `figures/idaes_surrogate_{A,B,C,D}.png` |
| `optimization/ipps_idaes.py` | IPPS MILP with the reluMIP nets (A, B, D), same costs/wind/grid modes as the original IPPS; `--days N` short horizon, `--urea-flex`, writes + replots the op point after each mode, reports which inputs sit on the data-box edge |

### Sections and data
- **A ammonia** (ammonia_flowsheet, case-2 settings): inputs Ft, Fh2 -> aq. NH3, water, electricity (no electrolyser). Only Fh2 = 10, Ft = 8500 converges cold; everything else is continuation. **Steps away from the base need Ipopt 3.14** (3.13 fails at the first 50 kg/h Ft step). 55 points, Ft 7500-9080, Fh2 ~7.5/8-15. Below that the reactor is H2-starved and fails (TODO A4). NH3 = min(~1.22 Fh2, nitrate limit); above Fh2 ~ 10 the extra H2 is wasted.
- **B NH3 recovery** (shortcut S2, algebraic): inputs = A outputs. Electricity ~148-169 kW, set by the 5000 kg/h water, almost independent of NH3.
- **C urea, ureaF replica**: 57 points, Fnh3 8-12, Fco2 10-50 (continuation from 10/15). **It never exceeds ~0.5 kg/h urea at <5 wt%**; urea falls as Fco2 rises, the opposite of Aspen (Fco2 ~44 -> 5.6 kg/h at 39 wt%). It cannot meet the IPPS demand (5.2 kg/h at >= 32.5 wt%), so it is trained and plotted but **not used in the IPPS**. The replica's high-CO2 behaviour needs fixing (TODO C).
- **D low-pressure DEF plant** (the IDAES plant's urea route) replaces C in the IPPS: input Fco2 -> urea, NH3 make-up, electricity; DEF is 32.5 wt% by construction. Converges cold only at Fco2 >= 14; continuation (Ipopt 3.14, adaptive step) reaches **8.5-20 kg/h** (9 points, ~linear). Even at 8.5 it makes 11.6 kg/h urea, 2.2x the IPPS demand, so urea-always-on overproduces (`--urea-flex` lets it cycle).

Fits (first round): R² >= 0.99 for every output, reluMIP vs Keras within 2e-4.

### IPPS result so far (first data box, 888 h, free_grid)
Optimum at the **lower corner of every box**: Ft 8500, Fh2 9.0, DEF Fco2 10.0.
W 1125 kW, battery 67 kWh, electrolyser 459 kW, H2 store 111 kg, NH3 store 444 kg,
ammonia train on 73 % of hours, 42 kWh/kg urea, £1.11/kg urea excl. the plant-capex
intercept (£121/kg with it). grid_10pct was similar. These numbers are from the
first box and are superseded once the expanded data is used.

### Where it stopped and what is next
1. Domain expanded downward after that optimum: A to Ft 7500 / Fh2 ~7.5, D to Fco2 8.5. Retraining was interrupted: **the A nets are retrained on the new data, but B, C, D nets and `metadata.json` are from the first round** -> rerun `python surrogates/idaes_train_relumip.py` before using them.
2. Run `python optimization/ipps_idaes.py --days 10` (operating-point learning), look at the edge report and the replotted surfaces, extend the data toward that edge with `idaes_data.py ... --ft/--fh2/--fco2`, retrain, repeat until the optimum is interior; then run the full 888 h.
3. Ft and Fh2 will probably keep going down (less brine, less H2), limited by H2 starvation in A (A4) and by D converging only down to Fco2 8.5. A DEF plant sized for 5.2 kg/h urea needs Fco2 ~3.8: needs a better continuation path (e.g. scale every DEF flow together) or a smaller-plant spec.
4. Production constraint: like the original IPPS, total urea over the horizon >= demand; there is no per-day minimum. With always-on urea that is moot; add a daily minimum if `--urea-flex` is used.
5. Fix the ureaF replica (C) high-CO2 behaviour, then compare C and D.

### 10-day IPPS on the (8,) nets (2026-11-27)
`python optimization/ipps_idaes.py --days 10` (240 h, Gurobi here, HiGHS if
installed), schedules `figures/ipps_idaes_schedule_10d_<mode>.png`
(`ipps_idaes_schedule_plot.py _10d`), op points on `figures/idaes_surrogate_*.png`.

| | free_grid | grid_10pct | grid_5pct | wind_only |
|---|---|---|---|---|
| Ft / Fh2 [kg/h] | 7743 / 7.56 | 7999 / 8.37 | 7999 / 8.37 | 7743 / 7.56 |
| DEF Fco2 / urea [kg/h] | 3.63 / 5.21 | same | same | same |
| wind kW / battery kWh | 326 / 267 | 346 / 140 | 346 / 140 | 343 / 223 |
| electrolyser kW / H2 store kg / NH3 store kg | 386 / 30 / 112 | 427 / 37 / 145 | same | 386 / 20 / 85 |
| ammonia train on-time | 32 % | 29 % | 29 % | 32 % |
| kWh/kg urea | 41.9 | 44.5 | 44.5 | 44.1 |
| £/kg urea excl. plant intercept | 1.96 | 2.14 | 2.14 | 1.96 |

Reading: the DEF plant runs continuously at exactly the demand (interior of
D, 3-20). The ammonia train runs ~30 % of hours at ~9.5 kg/h NH3, mostly in
windy hours, and the NH3 tank (85-145 kg) carries the DEF plant between
runs; the battery covers short gaps. Fh2 sits on the **low edge** of A in
free_grid and wind_only, where A has only 8 data points (Fh2 < 8.6) and the
reactor is H2-limited (NH3 ~ 1.22 Fh2). B's op point is interior.

### Proposed next data domain
- **A:** Ft 7000-8500 (step 250) x Fh2 5.0-9.0 (step 0.25), plus the existing
  points. Walk on Ipopt 3.14 with Fh2 steps of 0.05; expect convergence
  failures below ~7.5 (H2-starved reactor, TODO A4), so this is also the
  test of A4.
- **B:** shrink to the region used, NH3 5-12 kg/h x water 4500-5200 kg/h,
  40x40 (denser where the optimum is).
- **D:** demand pins Fco2 at 3.63; densify 3-6 kg/h at 0.1 steps; 6-20 can stay.
Then retrain (8,) nets and rerun the 10-day IPPS; stop when no input is on an edge.
