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
