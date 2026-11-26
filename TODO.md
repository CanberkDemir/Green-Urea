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

### A2 `[DONE]` Kinetic network choice: use the `.bkp`
- **What:** the `.bkp` reaction network is the correct basis for reproducing the existing Aspen case and gives ~55% NH3 selectivity. This does not validate its pseudo-first-order rate laws as physical Pd–Cu kinetics. The paper (Eqs. for r1–r4) and SI Table S7 instead describe a different network giving ~90%; those descriptions need correction and are not the Aspen model specification.
- **Why it matters:** NH3 yield, H2 demand and everything downstream.
- **Where:** `reactions/ammonia_reactions.py` (`NETWORK = "bkp" | "si"`), `docs/references.md` §1.
- **Notes:**
  - Done 2026-11-25: both networks are implemented and switchable; the default is still `"bkp"`.
    - **"bkp":** 55.4% selectivity, 11.96 kg/h NH3, ~9.8 kg/h H2 at full conversion.
    - **"si":** 87.3% selectivity, 18.85 kg/h NH3, ~11 kg/h H2 at full conversion (at Fh2 = 10 it is H2-limited and hits the A4 solver limit).
  - **Literature (R1–R4):** Pd–Cu is an N2-selective drinking-water catalyst, and in alkaline media it largely stops at nitrite. 90% NH3 is not supported for Pd–Cu. Ru/C is NH3-selective with no nitrite intermediate, including in waste ion-exchange brines (Huo et al. 2020, coupled with membrane distillation, which is the closest analogue to our process).
  - **Historical-model decision:** retain the `.bkp` network and its ~55% selectivity only as the Aspen-replication benchmark. The physical reactor design is moving forward with Ni₂P (A4); do not transfer `.bkp` kinetics or selectivity to Ni₂P. Align manuscript/SI descriptions and historical yield and energy claims with what Aspen actually ran.
  - **Recycle boundary:** the default IDAES case already uses the ~55% `.bkp` network and recycles 90% of the flash gas (mainly H2). NH3 is chiefly in the liquid product. Recovering or recycling NH3 can reduce product losses or improve its use in the downstream urea section, but it cannot raise nitrate-to-NH3 selectivity from ~55% to ~80%: nitrate converted to N2 is unavailable to that loop. Report reactor selectivity, NH3 recovery, and overall NH3-to-urea utilization separately.
  - Also check the paper's Zhang et al. reference (`\cite{Pd-Cu_experimental}`).
### A3 `[DONE]` H2 recycle: the paper is wrong; the lever is the fresh-H2 feed, not the recycle fraction
- **What:** the paper (§Energy-Demand Structure, lines ~1014–1029, and the LCA lines ~1254–1260) says the roughly 2× stoichiometric H2 excess "is not recycled" and that recycling it would cut electrolysis from about 77 to about 20 kWh/kg urea.
- **Why it matters:** electrolysis is about 40% of plant electricity.
- **Where:** `ammonia_flowsheet.py` (`RECYCLE_FRAC`, `h2_feed`); sweep run on 2026-11-25.
- **Notes:**
  - **The "excess" is not excess.** With the `.bkp` network, H2 consumption at full nitrate conversion is 9.8 kg/h: 6.4 kg/h goes to the NH3 route (4.5 H2/NH3) and 3.4 kg/h to the N2 side reactions (3 H2/N, ~45% of nitrate). The paper's "4.4 kg/h stoichiometric" counts the NH3 route only.
  - **The recycle already exists.** The model recycles 90% of the flash gas at 27 bar, so no recompression is needed.
  - **H2 lost in the purge = fresh − consumed, independent of the recycle fraction.** Swept 0.80/0.90/0.95/0.98 × Fh2 9.5–10.5 kg/h: at Fh2 = 10 the loss is 0.21 kg/h at every recycle fraction, and at 10.5 it is 0.71 kg/h. A higher recycle only grows the loop (0.43 → 5.25 mol/s) and C1 cooling (71 → 79 kW).
  - **Decision:** control fresh H2 at the consumption requirement (~9.8 kg/h at Ft = 8500, i.e. ~0.82 kg H2/kg NH3) with a small margin, and keep the recycle at 0.8–0.9. This saves at most ~2% of electrolysis compared with Fh2 = 10.
  - **Real lever:** selectivity (A2). At ~90% selectivity the same nitrate gives ~19.4 kg/h NH3 at ~0.57 kg H2/kg NH3, which is 30% less electrolysis per kg NH3.
  - **Paper text to fix:** "not recycled" and the 77 → 20 kWh/kg claim.
### A4 `[WIP]` Ni₂P reactor kinetics and H₂ delivery
- **Dossier implementation (2026-11-26):** `ni2p_dossier.py` adds Table S1 measurements/apparent fit, NaCl H2 solubility, H2 diffusivity and the transcribed Iliuta transfer correlation. The PFR uses the fitted rate, 12.39% equivalent loading and new solubility: default 1.25% conversion, 0.259 kg/h NH3 equivalent. Four conservation/limiting-case tests pass; solubility/diffusivity reproduce dossier examples. See `docs/A4_dossier_implementation.md` for scope and remaining work. Fixed selectivity, pH, hydrodynamics and high-concentration kinetic extrapolation remain unvalidated; A4 is not closed.
- **Catalyst decision (2026-11-26):** **Ni₂P is the catalyst selected for all forward-looking nitrate-to-ammonia reactor work.** Pd–Cu and the `.bkp` reaction set remain historical comparison cases, not the physical design basis. `docs/A4_Ni2P_PFR_Parameters.md` records a provisional Ni₂P basis from Wei et al. (2020): unsupported nanocrystals, 60 °C, ~1 atm total pressure, 5–10% H₂ in Ar, initial pH 2, ~96% NH₃ selectivity, and an approximate batch-derived nitrate disappearance coefficient of 1.09 h⁻¹. The proposed SiO₂-supported pellets, 12.2 wt% loading, transport coefficients, and 27 bar operation are unvalidated assumptions. Its single 1 m diameter × 10 m bed differs from the historical flowsheet's 15 parallel tubes; settle geometry before using catalyst inventory for design.
- **What:** the `.bkp` rates are zero-order in H2, so the model could consume H2 it doesn't have and went infeasible below ~9.8 kg/h.
- **Why it matters:** the optimizer must be able to run with less H2. All available H2 should react, and the leftover nitrate should pass through unreacted.
- **Where:** `reactions/ammonia_reactions.py` (`power_law_h2_limited`, `K_H2`, `H2_LIMITED`), `ammonia_flowsheet.py` (`solve_h2_continuation`), `test_kinetics.py`.
- **Notes:**
  - Done 2026-11-25: every rate is multiplied by C_H2/(C_H2 + K_H2) with K_H2 = 20 mol/m3 (~1% of the inlet H2).
  - Results unchanged at Fh2 ≥ 10. At 9.75 / 9.5 / 9.25 kg/h, H2 leaves at ~1e-11 mol/s, NH3 is 11.91 / 11.60 / 11.30 kg/h, and NO3- out is 0.0015 / 0.0105 / 0.0195 mol/s. This is the intended behaviour.
  - K_H2 = 1 was too sharp to solve.
  - **Still open:** cold starts fail below ~9 kg/h, and warm continuation is unreliable (IPOPT restoration). Robust init for the strongly starved regime is untested. Consider more PFR elements or a dedicated initializer.
  - `H2_LIMITED = False` restores the verbatim `.bkp` kinetics.
  - **Physical kinetics follow-up:** the old common H2 saturation factor is a numerical availability approximation, not a fitted Ni₂P rate law. Keep the `.bkp` case as the Aspen benchmark; fit the separate Ni₂P case when adsorption, rate, and product-selectivity data over the relevant dissolved-H2, nitrate/nitrite, pH, temperature, and brine ranges are available.
  - **Literature implementation:** `reactions/pdcu_pintar.py` implements Pintar et al. (1996) Eq. (3) and Table 2 as a separate nitrate-disappearance calculator; `test_pdcu_pintar.py` reproduces the paper's Table 3 calculated initial rate at 293 K. It is not connected to the IDAES PFR because the paper has no NH3/N2 branching parameters, and its dilute-water, <=293 K, <=0.46 bar H2 domain does not cover the 333 K, 27 bar brine case. The IDAES liquid-only reactor also lacks a gas-liquid H2 transfer model. A4 remains open pending a complete product network, catalyst loading, gas-liquid H2 treatment, and robust low-H2 PFR solves.
  - **Separate physical screening model (2026-11-26):** `idaes_model/h2_transfer_reactor.py` now tracks gas H2, dissolved H2, gas-liquid transfer, a liquid-catalyst H2 film, and balanced nitrate/nitrite-to-NH3/N2 pathways. At 60 C and 27 bar on the un-concentrated feed, its scenario solubility is ~12 mol/m3 (24 mg/L), so a saturated liquid outlet carries ~0.20 kg/h H2 without reaction; the rest of a 10 kg/h feed remains gas. The NaCl salting-out coefficient, kLa, film transfer, kinetic prefactors, and NH3 branching are exposed assumptions, not fitted values. The 1,400 kg/h-water B6 concentrate is rejected as nonphysical single-phase brine. See `idaes_model/H2_TRANSFER.md` and `test_h2_transfer_reactor.py`. The existing IDAES PFR remains an Aspen replica; fitting physical rates and integrating a three-phase reactor remain open.
  - **Runnable Ni₂P PFR (2026-11-26):** `idaes_model/ni2p_pfr.py` uses the provisional single-bed geometry and active Ni₂P inventory from `docs/A4_Ni2P_PFR_Parameters.md`, with local gas/liquid H₂ transfer, catalyst-surface H₂, and a balanced lumped NO₃⁻→NH₃/N₂ network. It uses the document's batch-derived nitrate rate and 96% reference NH₃ selectivity, with an assumed H₂ saturation factor normalized at the lab reference pressure. The 27 bar, `kLa=0.001 s⁻¹` single-bed case gives 1.07% nitrate conversion and 0.222 kg/h NH₃; `kLa=0.01 s⁻¹` gives 6.29% and 1.305 kg/h. These are scenario outputs, not validated predictions. Gas H₂ dilution by generated N₂ is included. Ten transport/PFR tests pass. The existing IDAES PFR has not been replaced; A4 stays WIP until supported-catalyst activity, brine/pH and H₂ dependence, transfer coefficients, and nitrite/product-network data are validated.
  - **HCl stream (2026-11-26):** `ni2p_pfr.py` reports HCl to set pH 2 (Pitzer, HSO4- buffering) plus reaction protons: 3.9 + 1.1 = 5.0 kg/h at the default case; ~94 kg/h HCl and ~55 kg/h NaOH (pH 11 for stripping) at full conversion. Not yet in the IDAES flowsheet.
  - **Latest diagnosis (2026-11-26):** at 10 kg/h H₂ feed, full conversion to the assumed 96% NH₃ branch requires ~10.07 kg/h H₂, but the default `kLa=0.001 s⁻¹`, ~12 mol/m³ equilibrium solubility, and 1.571 m³ liquid holdup give an inlet transfer ceiling of only ~0.137 kg/h; ~0.108 kg/h actually reacts. The catalyst-surface H₂ concentration is nearly zero. In sensitivity runs, `kLa/ksa` of 0.01/0.01, 0.01/0.1, and 0.1/1 s⁻¹ give ~6.29%, ~12.18%, and ~99.29% nitrate conversion, respectively; these are uncalibrated scenarios. Raising catalyst activity alone or increasing fresh H₂ flow at fixed pressure and fixed transfer coefficients does not remove this bottleneck. More gas circulation might improve real contact, but the model does not yet link gas velocity to `kLa`, wetting, or pressure drop. **Next:** estimate/measure those transport relations and supported-Ni₂P kinetics, then revisit H₂ flow and recycle together. A4 remains WIP.
### A5 `[WIP]` Electrolyte thermodynamics (ELECNRTL)
- **What:** the paper uses ELECNRTL. IDAES uses a reduced eNRTL / ideal-plus-Henry package, so heats of mixing are missing.
- **Why it matters:** this explains the H1/C1 duty errors in A1, and those duties feed the energy breakdown.
- **Where:** `idaes_model/properties/enrtl_ammonia.py`.
- **Notes:** Aspen's APV140 parameters are licensed and cannot be extracted.
  - **Pitzer model (2026-11-26):** `brine_electrolyte.py` (open parameters): a_w, osmotic pressure, BPE, NaCl saturation index, NH4+/HSO4- speciation, HCl/NaOH dosing. Matches NaCl data to 6 m within 0.005; 5 tests in `test_brine_electrolyte.py`. Used by `ni2p_pfr.py` (HCl) and `water_removal.py`. **Still missing:** excess enthalpy (the H1/C1 heat-of-mixing gap), T-dependence beyond NaCl. See `docs/A5_concentration_and_water_removal.md`.

### A6 `[DONE]` Cooling utilities for C1 and reactor loop
- **What:** cooling water serves C1 at the 40 °C design flash, the isothermal reactor, and the intercoolers. An R717 cycle serves only the optional C1 duty below the 35 °C cooling-water limit.
- **Why it matters:** the default flash now needs no chiller electricity; water-cooling pump/fan electricity is estimated separately.
- **Where:** `idaes_model/cooling_utilities.py`, `refrigeration.py`, `ammonia_flowsheet.py`, `ammonia_energy.py`.
- **Notes:**
  - Implemented 2026-11-26. Cooling-water supply is 30 °C with 5 K approach; the water-to-chiller split for colder C1 settings uses constant heat capacity. The R717 cycle retains 5 °C evaporation and 35 °C condensation. At the default 40 °C flash its circulation and power solve to zero.
  - Allocation checks run for 10, 30, 40, and 50 °C outlets. The 40 °C duty is 100% cooling water; at 10 °C the temperature-based estimate is 50% water and 50% chiller.
  - The full IDAES flowsheet was not rerun after this change because IDAES, Pyomo, and CoolProp are unavailable in the present environment. Chiller sizing below 35 °C needs an enthalpy-based split and heat-exchanger design.

### A7 `[DONE]` Ammonia-side energy breakdown
- **What:** an equipment-level table of the ammonia section's duties (heating, cooling, shaft work) and electricity at realistic COPs, taken from the IDAES model.
- **Why it matters:** the starting point for "reduce energy from the root", and it replaces the paper's sum-of-absolute-duties convention.
- **Where:** `idaes_model/ammonia_energy.py`.
- **Notes:**
  - Done 2026-11-25. At Ft = 8500 and Fh2 = 10: reactor 236.7 kW (hot, 60 C), C1 72.2 kW (60 → 10 C), H1 41.9 kW (29 → 60 C), intercoolers 13 kW, H2 compression 18.7 kW.
  - Paper convention (sum of absolute duties) = 383 kW-eq.
  - Electrolysis is not included (see F1).

### A8 `[DONE]` Flash B2 at 40 C
- **What:** the IDAES design default now cools reactor effluent to 40 °C before B2; the original `.bkp` specified 10 °C.
- **Why it matters:** lowering the cooling need at the source.
- **Where:** `ammonia_flowsheet.py` (`T_FLASH`), `ammonia_energy.py --tflash`.
- **Notes:**
  - Tested 2026-11-25: 10 / 30 / 40 / 50 C all give 11.96 kg/h NH3. C1 duty falls 72 → 43 → 29 → 15 kW, and it can use cooling water instead of the chiller.
  - Re-checked with the temperature-dependent NH3 Henry constant (A10): NH3 lost to the purge is only 0.001 → 0.004 kg/h (0.03%) at 50 C, because the gas recycle (~1 mol/s) is tiny compared with the liquid (~130 mol/s).
  - **Decision implemented 2026-11-26:** 40 °C is the design default; `--tflash 10` remains available for comparison. The 40 °C result in the 2026-11-25 sweep gave about 29 kW C1 duty, all assignable to cooling water with the 30 °C supply / 5 K approach assumption.
  - The sweep used the legacy ammonia flowsheet, not the separate Ni₂P PFR. Product-rate and NH₃-recovery claims for a combined Ni₂P-plus-flash model remain open under A4; no combined-model result is implied here.
### A9 `[DECIDE]` Brine pre-concentration (B6): infeasible as modelled, and the single largest hidden load
- **Decision deferred (2026-11-26):** leave A9 open for further discussion and continue development with two explicit alternatives: (1) remove water upstream using a physically specified concentration process and include its energy demand; (2) send unconcentrated or less-concentrated brine through the reactor and include the higher downstream NH3 recovery duty. Partial concentration remains a comparison case. Neither route is selected as the final design. Compare total process energy with heat integration and consistent NH3 recovery/product purity; do not treat the historical 1,400 kg/h-water B6 outlet as a feasible single-phase feed or omit water-removal energy.
- **What:** B6 removes 6.2 t/h of water at zero energy and leaves a 38.6 wt% salt concentrate. That is above NaCl saturation (56 g NaCl per 100 g water vs ~36), so ~280 kg/h NaCl would crystallize. The feed (103 g/kg TDS) is already above the conventional RO limit, so the water removal is thermal (MVC ~24 kWh/m3, crystallizer ~35 kWh/m3; refs R7).
- **Why it matters:** about 218 kW at the `.bkp` point, about as much as all other ammonia-section loads together.
- **Where:** `ammonia_energy.py` (`brine_concentration`, `--s16water`, `--no-b6-energy`), `ammonia_flowsheet.py` (`b6.s16_h2o_target`, now a mutable Param for continuation).
- **Notes:** sweep run 2026-11-25. Ammonia-section kWh/kg NH3 (heat-integrated, heat-pump heating, including B6):

  | Water left in concentrate | Membrane | Vacuum stripper |
  |---|---|---|
  | 1,400 kg/h (`.bkp`) | 29.2 | 23.7 |
  | 4,000 kg/h | 19.1 | 14.8 |
  | 5,000 kg/h | 17.4 | **14.6** |
  | 6,000 kg/h | 15.6 | 17.2 |
  | 7,600 kg/h (no concentration) | **12.9** | 21.3 |

  - The best design depends on the recovery technology (B2/B3): no concentration with the membrane, or concentrate to ~15–18 wt% salts with the vacuum stripper. Either is about 2× better than the `.bkp`.
  - NH3 product is unchanged across the range (the reactor is oversized).
  - The 2,637 and 3,000 kg/h points are numerically fragile.
  - Default left at 1,400 pending the decision.
  - **RO replacement screening (2026-11-26):** see `docs/B6_RO_screening.md` and `idaes_model/ro_screening.py`. At 25 °C, reference osmotic pressure is ~93 bar. A 120 bar unit removes only ~350 kg/h water with the stated driving-pressure/polarization assumptions; leaving 5000 kg/h requires ~169 bar (153–200 bar osmotic-coefficient sensitivity). Reference 120 bar pumping is ~33 kWe without hydraulic energy recovery, ~18 kWe with recovery to 27 bar. Vacuum recovery heat falls only ~309→295 kWth at fixed NH3 production. This is a screening calculation run in `idaes-env`, not an IDAES membrane simulation; importing IDAES in that environment currently fails. A9 remains open.
  - **Staged removal with Pitzer (2026-11-26):** `water_removal.py`, `docs/A5_concentration_and_water_removal.md`. RO at 120 bar removes only ~7%; practical NaCl floor ~2,500 kg/h water after NaOH. Removal saves 39 kWh_th/m3 of stripper heat; MVC costs ~18 kWh_e/m3, so it pays only if stripper heat is not heat-pumped or free. If used, remove water downstream on the acidic (pH 2) effluent: same energy, no H2-solubility penalty (upstream gives 0.57-0.63x), NH3 loss ~0.2 g/h.
  - **B6 bypass screening (2026-11-26):** `compare_b6_recovery.py` exercises the existing recovery shortcuts at fixed NH3 production and 40 °C feed; results in `docs/B6_recovery_screening.csv`. At 11.96 kg/h NH3, bypassing B6 raises atmospheric recovery hot utility after internal heat integration from 111 to 558 kWth, or vacuum recovery from 64 to 309 kWth, while avoiding the estimated 218 kWe B6 load. These are different energy forms and cannot be subtracted without a heating-utility model. No reactor heat credit, cooling electricity, or vacuum auxiliaries are included in these figures. The membrane constant-energy proxy was excluded because it cannot resolve dilution effects. The 1,400 kg/h case is an infeasible-liquid historical comparator. This is a controlled recovery sensitivity, not a full IDAES bypass solve or validated Ni2P process comparison; IDAES dependencies are unavailable here. Bypass is not established as the lowest-energy choice.

### A10 `[DONE]` Temperature-dependent NH3 Henry constant in the ammonia package
- **What:** NH3 Henry volatility was a constant Kpx = 6.0e4 Pa (Sander value at ~15 C). Stripping at 40–100 C and the warmer flash (A8) need its temperature dependence.
- **Why it matters:** NH3 volatility at 50 C is about 4.7× the constant value. The stripper and flash results would be wrong without it.
- **Where:** `properties/enrtl_ammonia.py` (`NH3_HENRY_T_DEPENDENT`, uses `VantHoffH` from `ideal_urea.py`).
- **Notes:**
  - Done 2026-11-25: van't Hoff with Sander (2015) H = 59 mol/(kg·bar) at 25 C and 4200 K, giving Kpx 9.41e4 Pa at 25 C, 5.8e4 at 15 C and 2.8e5 at 50 C.
  - The Aspen validation at 10 C is unchanged (NH3 −6.2%, Mt −0.04%).
  - Set the flag to False to restore the constant value.
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

### B3 `[WIP]` Column-based recovery: vacuum stripper validated rigorously; rectifier still shortcut
- **What:** a stripper (vacuum, driven by the 60 C reactor heat, or with MVR) and a rectifier that delivers dry NH3 to the urea section.
- **Why it matters:** a defensible alternative to the membrane (B2) and the real `Q_S`.
- **Where:** `ammonia_energy.py` (`recovery_streams`, shortcut), `nh3_stripper.py` (rigorous stripper).
- **Notes:**
  - Shortcut (2026-11-25): Kremser stripping, 10 stages, 99% recovery, Henry K from Sander; rectifier at 15 bar with Underwood Rmin (alpha = 6), R = 1.5 Rmin.
  - **Rigorous stripper (2026-11-25), `nh3_stripper.py`:** 10-stage MESH column (Mixer + Flash per stage) on the ammonia two-phase package, reboiler duty solved for 99% recovery.

    | Case | Reboiler | Overhead NH3 |
    |---|---|---|
    | 0.123 bar | 66.3 kW (shortcut 60.5) | 13.5 mol% (shortcut 13.8) |
    | 0.08 bar | 64.4 kW, stages 44–47 C | — |
    | Improved case (s16 = 5000, flash 50 C) | 190.8 kW (shortcut 208.7) | — |

  - The shortcut is within ~10% of the rigorous column.
  - **Correction from the rigorous model:** brine boiling-point elevation puts the stages ~6 K above pure water. At 0.123 bar the stages run at 56 C, only a 4 K approach to the 60 C reactor heat, which is infeasible. **Use 0.08 bar** (47 C, 13 K approach). `ammonia_energy.py` S2 was updated to 0.08 bar / 47 C. The improved case then gives an ammonia section of 7.4 kWh/kg urea and a plant total of 31.5.
  - **Still shortcut:** the 15 bar rectifier to anhydrous NH3, and the vacuum overhead condensation (10 C chiller assumed; NH3-rich vapour under vacuum).
  - Ideal liquid: no salting-out, which is conservative.
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


### C6 `[TODO]` ureaF converts only ~4% of the NH3 fed into urea
- **What:** in the Aspen grid data at Fnh3 = 9.96 and Fco2 = 14.58 kg/h, DEF-OUT carries 10.87 kg/h × 0.0628 = **0.68 kg/h urea**, which uses only ~0.39 kg/h NH3. The rest leaves through purges (PURGE1–3), S42 and vents. The low-pressure DEF plant (total recycle) converts essentially all NH3: 11.96 kg/h NH3 → 21 kg/h urea.
- **Why it matters:** this, more than the energy accounting, is why the paper's ~190 kWh/kg urea is so high. Every kWh of electrolysis and ammonia recovery is spread over very little urea.
- **Where:** `aspen/data/ureaF_results_live.csv`, `urea_flowsheet.py`, `def_lowp_flowsheet.py`.
- **Notes:** check the purge fractions in the `.bkp` (B9 / B19 0.9, B26 0.95) and the S42 stream. A higher recycle or the DEF route fixes it. Flag for the rewrite.
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

### D6 `[WIP]` DEF flowsheet does not converge on Windows / IPOPT 3.13.2 since commit 5fc8a93
- **What:** after the carbamate-loop changes (10% carbamate retention, MP/LP decomposers, `passes=1`, 0.8/0.2 damped tear), `def_lowp_flowsheet.py` fails at "closed loop (fixed X): infeasible" for `--direct` 0, 0.5 and 0.9 on this machine. The README reports it solving on Canberk's platform (Linux, IPOPT 3.14).
- **Why it matters:** blocks refreshing D2's numbers and any DEF-based plant totals.
- **Where:** `def_lowp_flowsheet.py` (`initialize`, `solve`).
- **Notes:**
  - Diagnosed 2026-11-25. With `passes=9`, tear passes 1–5 converge smoothly: recycle water creeps 0.389 → 0.470 mol/s, slowed by the 0.2 damping. On pass 6 a unit initialization fails silently and returns garbage (CARB 5.9 mol/s), after which the tear diverges (318 mol/s at pass 8).
  - `passes=4` reaches the closed-loop solve but IPOPT reports locally infeasible.
  - The pre-5fc8a93 version (e55ae7c) solved here.
  - Suggested fix: guard every `init()` result (reject and keep the previous state if a unit returns non-optimal), and use less damping or a Wegstein tear.
  - Left for Canberk, since it is their latest change. Not fixed here.
### D3 `[WIP]` Validate the reactor conversion correlation
- **What:** X = 0.64 + 0.05(N/C − 4) − 0.09(H/C − 0.5) is a trend assumption. The patent gives no conversion. The 10% carbamate retention is also assumed.
- **Why it matters:** recycle size, energy and the benefit of D2 all scale with it.
- **Where:** `def_lowp_flowsheet.py` (`conversion()`), `docs/references.md` R9.
- **Notes:**
  - Literature check 2026-11-25: the **directions** are confirmed (X rises with N/C, falls with H/C; equilibrium maximum near 190–195 C; Inoue et al. 1972). Industrial ranges are consistent (equilibrium up to about 80–86% CO2 basis).
  - The **coefficients are not verified**. The correlation papers (Inoue 1972; Brouwer 2009; Voskov 2016 on UreaKnowHow) are paywalled or login-only.
  - **Next:** get the Inoue equation and refit, or switch to the R01 kinetic/equilibrium model.
### D4 `[TODO]` Rigorous stripper and DEF product specs
- **What:**
  - The purification stripper and the condensate NH3 stripper are single stages.
  - Biuret, CO2/carbonate and metals (ISO 22241) are not modelled.
  - CO2 and LP-gas compression work is computed analytically, because the property package has no liquid entropy.
- **Why it matters:** product quality claims and energy numbers.
- **Where:** `def_lowp_flowsheet.py`.
- **Notes:** none yet.

### D5 `[WIP]` Compare against the ureaF replica on the same basis
- **What:** a like-for-like energy table: same urea output, same utility COPs.
- **Why it matters:** quantifies the claimed energy saving of the low-pressure route.
- **Where:** `plant_energy.py`.
- **Notes:**
  - End-to-end results 2026-11-25, in kWh electricity per kg urea (`python plant_energy.py --def-module <e55ae7c copy>`):

    | Case | Total |
    |---|---|
    | 1. Paper-like (`.bkp` kinetics, B6 to 1400 with energy, membrane, 10 C flash, no heat integration) | 41.2 (electrolysis 24.2, ammonia 15.6, urea 1.4) |
    | 2. Improved (B6 to 5000, vacuum stripper, 50 C flash, plant-wide heat integration) | 32.0 |
    | 3. Case 2 + SI/Ru-like selectivity | 22.7 |

  - All three are below the paper's ~190 and its LCA threshold of 56 kWh/kg.
  - The "plant-wide HI saving" printed for case 1 is not meaningful (no heat integration in that case).
  - The ureaF replica is not yet run through the same accounting; see C6.
  - Rerun 2026-11-25 after A10 (T-dependent Henry) and the 0.08 bar stripper: case 1 41.2 (unchanged); case 2 **31.5**; case 3 failed with an IPOPT internalSolverError in the ammonia solve (SI network at Fh2 = 11.5, s16 = 5000). Its last converged value, before those two changes, was 22.7. This is solver fragility (see A4/A9), not a physical result. Re-run case 3 after making the ammonia initialization more robust.
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

### E2 `[WIP]` Plant-wide heat integration (ammonia + urea + DEF)
- **What:** run E1 on all streams together.
- **Why it matters:** cross-section heat matches (e.g. urea condensers → ammonia stripper).
- **Where:** `plant_energy.py` (pinch over ammonia and DEF streams).
- **Notes:**
  - First run 2026-11-25: the plant-wide match saves an extra 18 kW (case 2) and 27 kW (case 3) over section-wise integration. The DEF condensers (90 C) cover part of the ammonia-side heat demand.
  - The DEF streams come from the e55ae7c DEF version because of D6.
  - Not yet: exchanger network design or area/cost.
### E3 `[DONE]` Electricity accounting consistent with the paper's surrogates
- **What:** the paper's unit electricity (E_N, E_U) is the sum of absolute duties at COP = 1. IDAES should report heating, cooling and shaft work separately.
- **Why it matters:** the paper's 190 kWh/kg urea headline and the LCA threshold of 56 kWh/kg both depend on this convention.
- **Where:** `ammonia_energy.py` (`electricity`), used by `plant_energy.py`.
- **Notes:**
  - Shaft work is counted as is.
  - Heating is by heat pump (0.5 Carnot from 25 C), with resistance heating also reported.
  - Cooling above 35 C uses cooling water at 2% of the duty; below 35 C it uses the R717 chiller COP (5.84).
## F. Missing plant blocks (in the paper, not in IDAES)

### F1 `[DONE]` Electrolyzer block
- **What:** the H2 demand model from the paper: Faradaic efficiency 0.98 and e_spec = 50 kWh/kg H2.
- **Why it matters:** about 40% of plant electricity.
- **Where:** `idaes_model/plant_energy.py` (`E_SPEC_H2`, `FE`).
- **Notes:** algebraic, as in the paper, and applied to the fresh-H2 feed of the ammonia flowsheet. It contributes 24.2 kWh/kg urea with `.bkp` kinetics and 17.6 with the SI network.
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
