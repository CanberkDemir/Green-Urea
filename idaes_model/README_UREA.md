# IDAES replica of the ureaF (Bosch-Meiser) subprocess

A first-principles IDAES/Pyomo flowsheet reproducing the Aspen Plus model
`aspen/ureaF.bkp` (urea synthesis per Pustjens & Van Den Tillaart,
US 10,759,745 B2, Stamicarbon: HP carbamate synthesis loop, MP/LP
carbamate decomposition trains with three gas recycles, steam stripping,
and DEF-product concentration). Companion to the ammoniaF replica in this
folder (`README.md`); extraction tables in `personal/urea_idaes_plan.md`.

## Files

| File | Purpose |
|---|---|
| `urea_flowsheet.py` | The ~40-unit flowsheet: build, initialize (tear passes + Aitken acceleration), solve, validate against `aspen/data/ureaF_results_live.csv` |
| `properties/ideal_urea.py` | Working property package: ideal gas + Raoult (H2O, NH3, plus .bkp-PLXANT Antoine fits) + van't Hoff Henry (CO2); UREA/CARB liquid-only |
| `properties/srk_urea.py` | Faithful SRK(+kij) replica of the .bkp's SR-POLAR basis - currently blocked by the `cubic_roots.so` platform issue documented in its docstring |
| `reactions/urea_reactions.py` | RXNTOT kinetic network (verbatim stoichiometry/rates, scaled pre-exponentials) + carbamate condensation equilibrium (RGibbs replacement) |
| `test_urea_kinetics.py` | Standalone verification: rate-constant conversions, R01 equilibrium-closure at two feeds, RGibbs-replacement conversion |

## Running

```bash
conda activate idaes-pse
python test_urea_kinetics.py        # kinetics checks
python urea_flowsheet.py            # default Fnh3=10, Fco2=15 kg/h
python urea_flowsheet.py 8 12       # other (Fnh3, Fco2) grid points
```

## Aspen block -> IDAES unit mapping

| Aspen block | Spec (from .bkp) | IDAES unit(s) |
|---|---|---|
| `NH3-IN` / `CO2-IN` / `STRIPH2O` | 34 degC/160 kgf/cm2; 100 degC/141 kgf/cm2; 60 kg/h steam 160 degC/5 bar | `Feed` x3 |
| `B5`, `B11`, `B14`, `B15`, `B24`, `CARBMIX2`, `M01` | mixers | `Mixer` (momentum none, level pressure fixed) |
| `NH3P1/NH3P2/B13/P1` | pumps to 20/200/200/5 bar | `Pump` (eta = 0.75 nominal) |
| `NH3SP1`, `B9`, `B19`, `B26` | splits 0.1 / 0.9 / 0.9 / 0.95 | `Separator` (totalFlow) |
| `B6`, `B27`, `B3` | heaters 200/200/20 degC | `Heater` |
| `B2/B22/X3/B23` | valves to 20/4/1/1 bar | `Heater` (Q=0, P fixed) |
| `R01` | RPlug, isothermal 200 degC, 141 kgf/cm2, 20 m x 2.5 m (grid geometry), reactions RXNTOT | `CSTR` (equilibrium-dominated; volume-homotopy init) |
| `F01/B1/F012/B8/B12/B17/B18/B4` | Flash2 at their T/P specs | `Flash` |
| `A02/A022` | RStoic, CARB -> 2NH3+CO2, conv 1.0 | `StoichiometricReactor`, extent pinned to inlet CARB |
| `A01/A012/A013` | RGibbs, 72.4 degC, UREA/N2/O2 inert | `StoichiometricReactor` + carbamate-equilibrium conversion spec (see reactions module) |
| `B10/B16/B20` | HeatX shortcut, hot out 40 degC vs dummy CW | hot-side `Heater` only (CW duty = -Q_hot) |
| `B7` | RadFrac 10 stages, partial-V condenser, 20 bar, D:F=0.9 | column registry: single-stage flash, V/F=0.9 |
| `B25` | RadFrac stripper, no condenser, QN=50,000 kJ/h, 5 bar | column registry: single-stage flash, bottoms T spec (surrogate knob) |
| `B28` | DISTL 5 stages, RR=1, 1 bar, D:F=0.8 | column registry: mole-balance spillover splitter |
| `D1` design spec | vary NH3SP1 split for 60 wt% urea in S5 | `apply_design_spec(m)` (optional constraint) |

## Column modularity (for the reactive-distillation upgrade)

The three columns are built through the `COLUMN_IMPLS` registry in
`urea_flowsheet.py`. An implementation supplies three builders returning
named port dicts; the flowsheet wires arcs only to those ports:

```
b7 : {"s32_in", "s17_in", "s27_out", "s34_out"}
b25: {"s5_in", "steam_in", "s16_out", "urea_out"}
b28: {"feed_in", "def_out", "s30_out"}
```

To add the reactive-distillation B7 (the .bkp defines the vapor-phase
equilibrium set `REACT1`, `2NH3 + CO2 <-> CARB`, for exactly that purpose -
though it is attached to no block in the shipped model), register e.g.
`COLUMN_IMPLS["reactive"]` with a TrayColumn-based builder honouring the
same ports and build with `build_model(..., column_impl="reactive")`.
Nothing else in the flowsheet changes.

## Faithful vs approximated (deviation record)

Replicated verbatim from the `.bkp`:
* topology (37 active blocks; B21 single-inlet mixer collapsed),
* every T/P/split/geometry spec (pressures decoded as kgf/cm2 where the
  file uses unit-row `<8>`),
* RXNTOT stoichiometry, activation energies, rate ratios (pre-exponential
  magnitudes scaled x1e-5 for numeric conditioning - the .bkp's own
  comment declares them arbitrary above the equilibrium threshold, and
  `test_urea_kinetics.py` asserts both rate-pair equilibria close),
* UREA/CARB critical constants, formation data, ideal-gas cp (CPIGDP
  refits), and PLXANT-derived volatility data.

Documented approximations:
* **Thermo basis**: ideal gas + Raoult/Henry instead of RKS+kij (the SRK
  package exists but the cubic external functions have no working
  aarch64 build). Largest expected effects: HP-loop fugacities, NH3
  volatility from dilute aqueous liquors (Raoult overestimates), and
  carbamate condensation extents.
* **RGibbs blocks**: conversion-form replacement (99% of the limiting gas
  reagent) - within ~0.3% of the van't Hoff equilibrium at all three
  units' conditions, see `reactions/urea_reactions.py`.
* **Columns**: single-stage surrogates honouring each column's .bkp spec;
  B25's bottoms temperature is an explicit calibration knob. Column
  duties (QB7/QB25/QB28 reb/cond) are not comparable until rigorous
  TrayColumn implementations replace the surrogates.
* **A01/A013 pressure**: run at their train levels; the .bkp's recorded
  141 kgf/cm2 there is inert in Aspen (single-phase RGibbs) and would
  only manufacture spurious phase boundaries here.
* **B5 mixing pressure**: 20 bar (the NH3P1 discharge) instead of 1 bar -
  same downstream states, avoids an intractable boiling-point flash;
  only the unvalidated NH3P1 pump work differs.
* N2 (declared but zero-flow in every Aspen stream) is omitted.

## Validation at Fnh3 = 10, Fco2 = 15 kg/h

The flowsheet initializes through all four recycle loops (9 sequential
tear passes with capped Aitken acceleration) and the simultaneous IPOPT
solve of the full ~40-unit model terminates **optimal**. Against the
nearest Aspen grid row (Fnh3 = 9.958, Fco2 = 14.583, duties MMkcal/h):

| Quantity | IDAES | Aspen | Deviation |
|---|---|---|---|
| Ft_UREA-OUT (DEF-OUT mass) | 11.90 kg/h | 10.87 kg/h | +9.5% |
| Wurea (DEF-OUT urea frac) | 0.0325 | 0.0628 | -48% |
| QR01 | -0.0004 | -0.0011 | sign/order agree |
| QB3 | -0.0005 | -0.0044 | sign agrees, 8x small |
| QB6 | +0.0003 | +0.0023 | sign agrees, 8x small |
| QB27 | +0.0019 | -0.0158 | sign differs |

Interpretation:

* **Structurally faithful and mass-balance sound**: topology, specs,
  kinetics network, recycle closure, and the DEF-product water/urea
  disposition (total product mass +9.5%).
* **Wurea -48%** = urea production 0.39 vs ~0.68 kg/h. The rate-pair
  equilibria (K_c = 1) are honoured exactly (see kinetics test); the gap
  comes from the VLE basis - the ideal+Raoult/Henry package holds less
  NH3/CO2 in the synthesis liquor than Aspen's RKS+kij, so the
  carbamate-then-urea equilibria sit lower. The `srk_urea` package is
  the fix-forward once cubic externals work on this platform.
* **Duty magnitudes (QB3/QB6/QB27, all < 30 kW absolute)** carry the
  mixing-enthalpy and column-surrogate approximations; same caveat class
  as the Qh1/Qc1 deviations documented for the ammonia replica. QB27's
  sign is dominated by how much recycle gas recondenses in M01 at the
  synthesis pressure, which the surrogate-B25 water split controls.
* A note for the manuscript, mirroring the SI-Table-S7 flag on the
  ammonia side: in the recorded Aspen results **DEF-OUT is the DISTL
  bottoms** (the urea-enriched 0.2F cut; Wurea up to 0.72 with median
  0.34), although the runner script labels it a distillate-side product.
  The .bkp M-codes that suggest otherwise are graphics indices - see the
  phase-routing note in `urea_flowsheet.py`.
