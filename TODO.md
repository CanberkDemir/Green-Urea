# IDAES model — running TODO

The running list of what the IDAES models (`idaes_model/`) still lack compared with the
manuscript (`main_acs.tex`) and the Aspen files (`aspen/`), and what has been done so far.
Update an item's status tag and notes whenever you work on it. Add new items at the end
of their section and give them the next free ID.

## How to read this file

Every item starts with a status tag:

| Tag | Meaning |
|---|---|
| `[DONE]` | Implemented and run. Results checked as described in the notes. |
| `[WIP]` | Being worked on. The notes say what exists and what is missing. |
| `[UNTESTED]` | Code exists but has not been run or validated. |
| `[BLOCKED]` | Cannot proceed. The notes name the blocker. |
| `[TODO]` | Not started. |
| `[DECIDE]` | Needs a team decision before work starts. |

Every item has the same fields:

- **What:** the gap, in one or two sentences.
- **Why it matters:** the effect on the paper's results.
- **Where:** the files involved.
- **Notes:** free text: status details, dates, numbers, "not tested yet", and so on.

---

## A. Ammonia section (nitrate hydrogenation)

### A1 `[DONE]` IDAES replica of `aspen/ammoniaF.bkp`
- **What:** a 9-unit flowsheet: H2 compressor train, brine pre-concentration, isothermal PFR, flash, gas recycle and let-down.
- **Why it matters:** this is the rigorous model behind the ammonia surrogate.
- **Where:** `idaes_model/ammonia_flowsheet.py`, `idaes_model/README.md`.
- **Notes:** checked against Aspen at Ft = 8500 kg/h and Fh2 = 10 kg/h. Mass balance is within 0.04% and NH3 within -6%. The Qh1 and Qc1 heater duties disagree because of mixing enthalpy; see the README. Re-run on Windows on 2026-11-24 and the results reproduce.

### A2 `[DECIDE]` Kinetic network: the paper and SI disagree with the `.bkp`
- **What:** the paper (Eqs. for r1–r4) and SI Table S7 describe side reactions that give about 90% NH3 selectivity. The `.bkp`, and therefore the IDAES model, uses a different side-reaction network that gives about 55–59% selectivity.
- **Why it matters:** NH3 yield, H2 demand and everything downstream depend on it. The paper states 85–92% selectivity, but the model that was actually run does not reach that.
- **Where:** `idaes_model/reactions/ammonia_reactions.py`, `main_acs.tex` §Ammonia Catalytic Hydrogenation, SI Table S7.
- **Notes:** decide which network is correct, then either fix the paper text or re-tune the kinetics.

### A3 `[DECIDE]` H2 recycle: the paper and the model disagree
- **What:** the paper (§Energy-Demand Structure) says the roughly 2× stoichiometric H2 excess "is not recycled". The `.bkp` and the IDAES model recycle 90% of the flash vapour (SPLIT1).
- **Why it matters:** electrolysis is about 40% of plant electricity. The paper names H2 recycle as the largest single efficiency lever, at about 77 → 20 kWh/kg urea.
- **Where:** `idaes_model/ammonia_flowsheet.py` (`split1`), `main_acs.tex` lines ~1014–1029.
- **Notes:** check what the surrogate data actually assumed before changing either one.

### A4 `[TODO]` Hydrogen-starved operating regime
- **What:** the rate laws are zero-order in H2, so the model goes infeasible for Fh2 below about 9.8 kg/h at Ft = 8500.
- **Why it matters:** the optimizer may want to run with lower H2 (A3), which the model cannot represent.
- **Where:** `idaes_model/reactions/ammonia_reactions.py`.
- **Notes:** possible fix: add an H2-availability factor C_H2/(C_H2+eps) to the rates. This deviates from the `.bkp` rate law.

### A5 `[TODO]` Electrolyte thermodynamics (ELECNRTL)
- **What:** the paper uses ELECNRTL. IDAES uses a reduced eNRTL / ideal-plus-Henry package, so heats of mixing are missing.
- **Why it matters:** this explains the H1/C1 duty errors in A1, and those duties feed the energy breakdown.
- **Where:** `idaes_model/properties/enrtl_ammonia.py`.
- **Notes:** Aspen's APV140 parameters are licensed and cannot be extracted.

### A6 `[WIP]` Refrigeration utility for C1
- **What:** a closed R717 vapour-compression cycle (CoolProp) carries the whole C1 cooling duty.
- **Why it matters:** the paper charges cooling at COP = 1. A real chiller COP lowers cooling electricity several-fold.
- **Where:** `idaes_model/refrigeration.py`, `ammonia_flowsheet.py` (Canberk, 2026-11-25).
- **Notes:**
  - Evaporates at 5 °C and condenses at 35 °C.
  - The reactor duty (~237 kW at 60 °C) and the 50 °C intercoolers are not assigned to any utility yet. The paper itself says cooling water would suffice for those.
  - Cooling-water precooling of C1 has not been evaluated.

### A7 `[DONE]` Ammonia-side energy breakdown
- **What:** an equipment-level table of the ammonia section's duties (heating, cooling, shaft work) and electricity at realistic COPs, taken from the IDAES model.
- **Why it matters:** the starting point for "reduce energy from the root", and it replaces the paper's sum-of-absolute-duties convention.
- **Where:** `idaes_model/ammonia_energy.py`.
- **Notes:**
  - Done 2026-11-25. At Ft = 8500 and Fh2 = 10: reactor 236.7 kW (hot, 60 C), C1 72.2 kW (60 → 10 C), H1 41.9 kW (29 → 60 C), intercoolers 13 kW, H2 compression 18.7 kW.
  - Paper convention (sum of absolute duties) = 383 kW-eq.
  - Electrolysis is not included (see F1).

### A8 `[DECIDE]` Flash B2 at 40–50 C instead of 10 C
- **What:** the `.bkp` cools the reactor effluent to 10 C (C1, chiller) before flash B2.
- **Why it matters:** lowering the cooling need at the source.
- **Where:** `ammonia_flowsheet.py` (`T_FLASH`), `ammonia_energy.py --tflash`.
- **Notes:**
  - Tested in IDAES on 2026-11-25: 10 / 40 / 50 C all give 11.96 kg/h NH3 and the same purge.
  - C1 duty falls 72 → 29 → 15 kW, and it can use cooling water instead of the chiller.
  - Caveat: the thermo is ideal + Henry, so NH3 losses to the gas recycle may be underpredicted. Check with rigorous thermo or data before adopting.

### A9 `[TODO]` Brine pre-concentration (B6) has no energy cost
- **What:** B6 removes about 7 t/h of water from the 8.5 t/h brine (to 1400 kg/h) as an ideal Separator, in both Aspen and IDAES, at zero energy.
- **Why it matters:** this is a real separation. Reverse osmosis at about 3–5 kWh/m3 is roughly 25–35 kW. Thermal evaporation would be MW-scale. Neither appears in the paper's energy totals.
- **Where:** `ammonia_flowsheet.py` (`b6`).
- **Notes:** decide on the technology (RO is likely, but the brine is ~10% salts at the inlet and ~40% at the outlet, which is above RO's practical limit). Then add an energy term.

## B. Ammonia separation / recovery (hydrous → anhydrous NH3)

### B1 `[TODO]` Ammonia recovery is missing from IDAES entirely
- **What:** S13 leaves as about 0.5 wt% NH3 in about 2.3 t/h of water. The paper recovers it with a gas-stripping membrane (Iddya et al. 2020) at a fixed 7.1 kWh/kg N (≈ 5.85 kWh/kg NH3). No IDAES unit performs this step, so the IDAES ammonia and urea models are not connected.
- **Why it matters:** this step is the paper's `Q_S`, the largest recoverable heat and one of the largest ammonia-section loads.
- **Where:** new script (planned).
- **Notes:** decide between keeping the membrane as a black box (B2) and adding columns (B3).

### B2 `[DECIDE]` Membrane assumption: keep it or replace it?
- **What:** the membrane is a literature energy factor, untested at this feed (dilute, saline, 60 °C).
- **Why it matters:** it sets `Q_S`. The paper also assumes 90–99% internal recovery of `Q_S` with no temperature-level check.
- **Where:** `main_acs.tex` §Ammonia Separation and §Heat Integration.
- **Notes:** compare against B3 before deciding.

### B3 `[WIP]` Column-based recovery: steam/vacuum stripper plus rectifier to anhydrous NH3
- **What:** model a stripper (possibly vacuum, driven by the 60 °C reactor heat, or with MVR) and a rectifier that delivers dry NH3 to the urea section. Report the energy per kg NH3.
- **Why it matters:** it gives a defensible alternative to B2 and the real `Q_S`.
- **Where:** `idaes_model/ammonia_energy.py` (`recovery_streams`).
- **Notes:**
  - **SHORTCUT ONLY, NOT RIGOROUS.** Kremser stripping (10 stages, 99% recovery, Henry K from Sander 2015, salting-out ignored); rectifier at 15 bar with Underwood Rmin (alpha = 6), R = 1.5 Rmin.
  - Result, ammonia section in kWh/kg NH3 with heat integration (dT_min = 10 K, 10 C flash):

    | Option | kWh/kg NH3 |
    |---|---|
    | Membrane (paper) | 11.1 |
    | Atmospheric steam stripper | 15.5 |
    | Steam stripper with MVR | 9.5 |
    | Vacuum stripper (0.123 bar, 50 C) | 5.9 |

  - The vacuum stripper's reboiler is driven by the 60 C reactor heat; only the rectifier reboiler (198 C, ~16 kW) needs external heat.
  - **Next:** rigorous IDAES column(s) to confirm; vacuum overhead condensing (10 C chiller assumed); check alpha.

## C. Urea section (ureaF replica, Bosch–Meiser)

### C1 `[DONE]` IDAES replica of `aspen/ureaF.bkp` converges with IPOPT
- **What:** about 40 units across four recycles.
- **Why it matters:** this is the rigorous model behind the urea surrogate.
- **Where:** `idaes_model/urea_flowsheet.py`.
- **Notes:**
  - Fixed on 2026-11-24 with an initialization fallback and an IPOPT warm restart.
  - DEF-OUT mass is +9.5% vs Aspen; Wurea is -48% (see C2).

### C2 `[WIP]` Urea production is half of Aspen's (Wurea -48%)
- **What:** the R01 equilibria close exactly, so the gap comes from the column stand-ins and the recycle water.
- **Why it matters:** urea yield and the DEF concentration.
- **Where:** `urea_flowsheet.py`, `README_UREA.md` (investigation notes).
- **Notes:**
  - The B25 single-stage knob is ill-conditioned: at 426 K urea falls, and at 430 K the solve is infeasible.
  - Needs an Aspen stream table (S5, S16, UREA-OUT, REC3) at Fnh3 = 10, Fco2 = 15.

### C3 `[BLOCKED]` Rigorous B25 stripper (`column_impl="trays_b25"`)
- **What:** a 10-stage MESH column exactly as specified in the `.bkp`.
- **Why it matters:** it replaces the ill-conditioned surrogate.
- **Where:** `urea_flowsheet.py` (`_build_b25_trays`).
- **Notes:**
  - It builds but does not converge.
  - Aspen's +13.9 kW reboiler is inconsistent with about 50 kg/h of water leaving in the bottoms.
  - Blocked on the Aspen stream table (see C2).

### C4 `[TODO]` Rigorous B7 and B28 columns
- **What:** B7 (10-stage absorber) and B28 (5-stage DISTL, RR = 1, 1 bar; the paper's "distillation column") are still single-stage stand-ins.
- **Why it matters:** the paper says this column is about 0.1 MW and dominates urea-section energy. The stand-in duties cannot be compared with that.
- **Where:** `urea_flowsheet.py` (`COLUMN_IMPLS`).
- **Notes:** reuse the tray builder from C3.

### C5 `[BLOCKED]` SR-POLAR / SRK thermodynamics
- **What:** the faithful SRK package exists, but IPOPT crashes (0xC0000005) on any two-phase SRK state in idaes-ext 3.4.2, including IDAES's own example. Peng-Robinson works; a PR variant fails at the 200 bar pump B13.
- **Why it matters:** high-pressure loop fugacities and liquid densities, which set the concentration-based rates.
- **Where:** `idaes_model/properties/srk_urea.py`.
- **Notes:**
  - Re-test after an idaes-ext upgrade.
  - PR is the fallback; it is less important than C2–C4.

## D. Low-pressure DEF plant (new design, US 10,759,745 B2)

### D1 `[DONE]` Flowsheet built from scratch
- **What:** a 150 bar reactor-only HP section, MP/LP dissociation, a 0.5 bar steam stripper, DEF 32.5 wt% finishing, and carbamate recovery.
- **Why it matters:** the proposed lower-energy urea route; water leaves with the product, so there is no recycle build-up.
- **Where:** `idaes_model/def_lowp_flowsheet.py`, `figures/def_pfd.*`.
- **Notes:**
  - Mass balance closes (residual < 5e-5 kg/h).
  - Carbamate split and decomposers were added by Canberk on 2026-11-25.

### D2 `[DONE]` Condensate NH3 stripper and the condensate-to-DEF option
- **What:** `--direct` sends condensate straight to DEF dilution; a 4 bar reboiled flash (`--boilup`) strips its NH3 first.
- **Why it matters:** less water in the reactor, so higher conversion and lower HP and heating loads.
- **Where:** `def_lowp_flowsheet.py`.
- **Notes:**
  - At 90% direct: 651 ppm NH3 (ISO limit 2000) and heating 53 → 34 kW.
  - These numbers predate Canberk's carbamate changes. Re-run to refresh them.

### D3 `[TODO]` Validate the reactor conversion correlation
- **What:** X = 0.64 + 0.05(N/C − 4) − 0.09(H/C − 0.5) is a trend assumption. The patent gives no conversion. The 10% carbamate retention is also assumed.
- **Why it matters:** recycle size, energy and the benefit of D2 all scale with it.
- **Where:** `def_lowp_flowsheet.py` (`conversion()`).
- **Notes:** replace with the R01 kinetics, or with literature equilibrium data (e.g. Inoue).

### D4 `[TODO]` Rigorous stripper and DEF product specs
- **What:**
  - The purification stripper and the condensate NH3 stripper are single stages.
  - Biuret, CO2/carbonate and metals (ISO 22241) are not modelled.
  - CO2 and LP-gas compression work is computed analytically, because the property package has no liquid entropy.
- **Why it matters:** product quality claims and energy numbers.
- **Where:** `def_lowp_flowsheet.py`.
- **Notes:** none yet.

### D5 `[TODO]` Compare against the ureaF replica on the same basis
- **What:** a like-for-like energy table: same urea output, same utility COPs.
- **Why it matters:** quantifies the claimed energy saving of the low-pressure route.
- **Where:** new comparison script.
- **Notes:** none yet.

## E. Utilities, heat integration and energy accounting

### E1 `[DONE]` Reusable pinch / heat-integration module
- **What:** a problem-table pinch algorithm over a list of hot and cold streams (supply T, target T, duty), returning the minimum hot and cold utility and the pinch temperature. Applied to the ammonia side first, then plant-wide.
- **Why it matters:** the paper assumes 90–99% recovery of `Q_S` without checking temperature levels.
- **Where:** new module (planned), `idaes_model/heat_integration.py`.
- **Notes:**
  - Done 2026-11-25. Checked against the Kemp textbook 4-stream example (hot 20, cold 60, shifted pinch 85 C: exact).
  - Applied to the ammonia side in `ammonia_energy.py`.
  - Grand composite curves are in `figures/ammonia_gcc_*.png`.
  - Generic `Stream(name, t_in, t_out, duty_kw, section=...)` list, ready for E2.
  - The finding that partly contradicts the paper's 90–99% recovery of `Q_S`: with the membrane, `Q_S` is electricity with no temperature, so there is nothing to recover. With columns, recovery is real but temperature-limited; see B3.

### E2 `[TODO]` Plant-wide heat integration (ammonia + urea + DEF)
- **What:** run E1 on all streams together.
- **Why it matters:** cross-section heat matches (e.g. urea condensers → ammonia stripper).
- **Where:** E1 module.
- **Notes:** after E1 and B3.

### E3 `[TODO]` Electricity accounting consistent with the paper's surrogates
- **What:** the paper's unit electricity (E_N, E_U) is the sum of absolute duties at COP = 1. IDAES should report heating (electric heater or heat pump COP), cooling (cooling water vs chiller COP) and shaft work separately.
- **Why it matters:** the paper's 190 kWh/kg urea headline and the LCA threshold of 56 kWh/kg both depend on this convention.
- **Where:** a new reporting helper used by all flowsheets.
- **Notes:** none yet.

## F. Missing plant blocks (in the paper, not in IDAES)

### F1 `[TODO]` Electrolyzer block
- **What:** the H2 demand model from the paper: Faradaic efficiency 0.98 and e_spec = 50 kWh/kg H2.
- **Why it matters:** about 40% of plant electricity.
- **Where:** none yet (algebraic model in the paper).
- **Notes:** simple enough as a Pyomo block or a post-processing calculation.

### F2 `[TODO]` Link the ammonia → urea interface
- **What:** feed the recovered NH3 (B1/B3) into the urea and DEF flowsheets, plus the CO2 source and feed conditioning.
- **Why it matters:** an end-to-end plant model.
- **Where:** none yet.
- **Notes:** depends on B1.

### F3 `[TODO]` Autothermal reactor and intermittent operation
- **What:** the paper adopts the Cheema & Krewer autothermal reactor for intermittent operation. IDAES uses an isothermal steady-state PFR.
- **Why it matters:** intermittency is central to the paper's optimization.
- **Where:** `ammonia_flowsheet.py`.
- **Notes:** possibly out of scope for steady-state IDAES; document the assumption if so.

### F4 `[TODO]` IDAES-based grid runner for surrogate training data
- **What:** the surrogates are trained on the Aspen grids (`aspen/run_*_grid.py`). There is no equivalent sweep script for the IDAES models.
- **Why it matters:** the surrogates need regenerating once the utility/energy targets change (Canberk's note in A6).
- **Where:** new script; output CSVs in the same format as `aspen/data/`.
- **Notes:** none yet.

### F5 `[TODO]` Equipment costing
- **What:** CAPEX in the paper comes from outside IDAES. IDAES has costing modules that could size and cost units directly.
- **Why it matters:** a consistent techno-economic analysis.
- **Where:** none yet.
- **Notes:** low priority.
