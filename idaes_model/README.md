# IDAES replica of the ammoniaF (nitrate-hydrogenation) subprocess

A first-principles IDAES/Pyomo flowsheet reproducing the Aspen Plus model
`aspen/ammoniaF.bkp` (catalytic nitrate-to-ammonia conversion: brine
pre-concentration, H2 compression, isothermal liquid-phase RPLUG reactor,
flash separation with gas recycle, product let-down).

> Folder is named `idaes_model/` (not `idaes/`) so it cannot shadow the
> installed `idaes` package on `sys.path`.

## Files

| File | Purpose |
|---|---|
| `ammonia_flowsheet.py` | The 9-unit flowsheet: build, initialize, solve, and validate against `aspen/data/ammoniaF_results_live.csv` |
| `properties/enrtl_ammonia.py` | Property package (3 variants from one `get_prop(phases)` call) |
| `reactions/ammonia_reactions.py` | The AMMONIA1 kinetic network, replicated verbatim from the `.bkp` |
| `test_kinetics.py` | Standalone kinetics verification (ODE branching + IDAES unit consistency) |

## Running

```bash
conda activate idaes-pse            # has idaes + ipopt (conda-forge)
python test_kinetics.py             # kinetics checks
python ammonia_flowsheet.py         # default validation point Ft=8500, Fh2=10
python ammonia_flowsheet.py 8500 12.5   # other (Ft, Fh2) grid points
```

## Aspen block -> IDAES unit mapping

| Aspen block | Spec (from .bkp) | IDAES unit(s) |
|---|---|---|
| `H2FEED` | pure H2, 25 degC, 1 bar | `Feed` (vapor package) |
| `MCOMPR1` | 4-stage isentropic, SEFF=0.72/stage, intercooled to 50 degC (stages 1-3), 27 bar out | `cmp1..cmp4` (`Compressor`) + `ic1..ic3` (`Heater`) |
| `NO3-IN` | 8500 kg/h brine, 25 degC, 27 bar; mass frac H2O .897 / Cl- .053 / Na+ .039 / NO3- .009 / SO4-- .002 | `Feed` (liquid package) |
| `B6` (Sep) | all solutes to S16; S16 water capped at 1400 kg/h | `Separator` (componentFlow) + explicit flow constraint |
| `MIXER1` | 27 bar | `Mixer` |
| `H1` | 60 degC, 27 bar | `Heater` |
| `B1` (RPlug) | T-SPEC isothermal at inlet T, NTUBE=15, L=10 m, D=1 m, NPHASE=1 (liquid), reactions AMMONIA1 | `PFR`, area = 15*pi/4 m2, T fixed along length |
| `C1` | 10 degC, 27 bar | `Heater` |
| `B2` (Flash2) | 10 degC, 27 bar | `Flash` (Q=0, dP=0 after C1) |
| `SPLIT1` | N2REC : PURGE1 = 0.9 : 0.1 | `Separator` (totalFlow) |
| `V1` | P-OUT = 1 bar | `Heater` with Q=0 (isenthalpic valve) |
| `S13` | hydrous ammonia product | `Product` |

Three instances of the property package bridge the Aspen phase specs
(`Translator` blocks in between): vapor-only for the compressor train,
liquid-only (eNRTL) for the reaction train (matching RPLUG NPHASE=1),
two-phase Ideal+Henry for the separation train.

## Kinetics: the .bkp is the ground truth, not the SI

The manuscript SI (Table S7) describes the side reactions as
`r3: NO3- -> N2 (k=8.2e3)` and `r4: NO2- -> N2 (k=2.0e2)`, which gives
~93% ultimate NH3 selectivity at 60 degC. **The Aspen file implements a
different network**: SIDE1 (k=8.2e3, Ea=11970) consumes NO2-, SIDE2
(k=2.0e2, Ea=8370) consumes NO3-, both with stoichiometric coefficient -2
per extent, plus a T-EXP=1 prefactor on every rate:

```
MAIN1: NO3- +   H2 ->  NO2- +  H2O     r = k1 T e^(-Ea1/RT) C_NO3
MAIN2: 2NO2- + 7H2 -> 2NH3  + 4H2O     r = k2 T e^(-Ea2/RT) C_NO2
SIDE1: 2NO2- + 4H2 ->  N2   + 4H2O     r = k3 T e^(-Ea3/RT) C_NO2
SIDE2: 2NO3- + 6H2 ->  N2   + 6H2O     r = k4 T e^(-Ea4/RT) C_NO3
```

That network gives ~55% analytic selectivity, and the recorded Aspen
results confirm it (NH3_out = 12.75 kg/h at Ft=8500/Fh2=10 = 59% realized
yield, vs ~20 kg/h that the SI network would produce). This model
implements the file. Note also that MAIN2/SIDE1/SIDE2 as written in Aspen
are not charge-balanced (no OH- production; Aspen's apparent-component
solution chemistry absorbs this) - they are replicated verbatim, so charge
bookkeeping drifts exactly as in Aspen's records. **Worth flagging for the
manuscript: SI Table S7's pathway labels do not match the model that was
actually run.**

## Validation at Ft=8500 kg/h, Fh2=10 kg/h

Against `aspen/data/ammoniaF_results_live.csv` (duties in MMkcal/h):

| Quantity | IDAES | Aspen | Deviation |
|---|---|---|---|
| NH3 in S13 | 11.96 kg/h | 12.75 kg/h | -6.2% |
| S13 total mass (Mt) | 2280.9 kg/h | 2281.7 kg/h | -0.04% |
| NH3 mass frac (Wnh3) | 0.0052 | 0.0056 | -6.1% |
| Qh1 (H1) | +0.0360 | -0.0306 | sign differs (see below) |
| Qc1 (C1) | -0.0621 | -0.0867 | +28% |
| Qr1 (B1) | -0.2035 | -0.1866 | -9% |
| Qcomp (MCOMPR1 QNET) | -0.01115 | -0.01117 | +0.2% |

Interpretation:

* **Structurally faithful and quantitatively close**: topology, operating
  conditions, reactor sizing, kinetic network, compressor train (0.2%),
  and overall mass balance (0.04%) match.
* **NH3 (-6%) / Qr1 (-9%)**: residual kinetic-basis differences - Aspen
  computes molarity from its ELECNRTL true-species density and
  re-speciates NO3-/NO2-/NH3/NH4+ through its solution chemistry; this
  model uses molecular NH3 and an ideal-mixing molar volume.
* **Qh1 (sign) / Qc1 (+28%)**: heats of mixing. The B6 step concentrates
  the brine from ~10 to ~40 wt% dissolved salts; Aspen's ELECNRTL enthalpy
  includes the (large) heat-of-solution effects, whereas this package uses
  ideal mixing of constant partial-molar aqueous species. The absolute
  offsets (~70 kW on ~2.3 t/h of concentrated brine) are consistent with
  electrolyte mixing-enthalpy magnitudes. These duties would need an
  excess-enthalpy-capable electrolyte package to close. But note the
  branch caveat below: on Aspen's own low-Fh2 branch (Fh2 <= 7.5) Aspen
  reports Qh1 = +0.056 to +0.058 - the *same sign and comparable
  magnitude* as this model - so part of the Fh2=10 "sign disagreement" is
  Aspen's branch switching, not thermodynamics.

## Caveat: the Aspen grid data itself has solution branches

Scanning `ammoniaF_results_live.csv` at Ft=8500 across Fh2 shows at least
three regimes in the *Aspen* results, with discontinuous jumps:

* **Fh2 <= ~7.5**: smooth, Qh1 positive (+0.056...+0.058 MMkcal/h),
  NH3_out tracks the H2-starved stoichiometric limit (all H2 consumed).
* **~7.8 <= Fh2 <= ~10.5**: Qh1 flips to about -0.03 with no physical
  reason at that scale; NH3_out rises to ~12.7 (this is the branch the
  Fh2=10 validation row sits on; NH3_out even spikes to 13.7 at
  Fh2=9.83).
* **Fh2 >= ~10.8**: Mt drops discontinuously from ~2281 to 1480.2 kg/h,
  Qc1 flips sign to +0.65...+0.69, Qr1 quintuples to ~-1.0, and NH3_out
  freezes at 11.87-11.88 kg/h regardless of Fh2.

These look like the sequential-modular recycle tear converging to
different fixed points across the automated grid (the repo history notes
the runs were time-limited). The IDAES model solves the recycle
simultaneously and lands on one consistent branch: its NH3 (11.96 kg/h)
matches Aspen's high-Fh2 plateau (11.87, +0.7%), its Mt matches the
low/mid branches (0.04%), and its Qh1 sign matches the low branch.
Treat single Aspen rows near the branch switches with caution.

## Applicability domain: H2-starved feeds are out of scope

The kinetic rates are zero-order in H2 (faithful to the .bkp EXPONENT
records), so hydrogen exhaustion cannot slow the reactions. Full NO3-
conversion at Ft=8500 consumes ~9.8 kg/h H2; running the model with
Fh2 below that makes the PFR hydrogen balance infeasible (Aspen's
marching integrator instead clips the reactions where H2 runs out along
the tube, which is how its low-Fh2 rows track the stoichiometric limit).
Use Fh2 >= ~10 kg/h at Ft=8500, or scale accordingly. Adding a smooth
H2-availability factor to the rate forms (e.g. C_H2/(C_H2+eps)) would
extend the model to the starved regime at the cost of deviating from the
file's documented rate law.

## Thermodynamics: what is and is not eNRTL here

Aspen uses ELECNRTL ("ENRTL-RK") with the licensed APV140 databank; the
binary-interaction values are referenced by bank name in the .bkp and are
not extractable. Choices made (all documented inline in
`properties/enrtl_ammonia.py`):

* The **liquid-only (reactor train) variant uses IDAES's eNRTL** with
  Chen (1982) parameters for (H2O, Na+ Cl-) and Aspen's documented
  defaults (8, -4) elsewhere. Note IDAES's eNRTL contributes activity
  coefficients (available on the state blocks) but not enthalpy or
  fugacity, which come from the inherited Ideal EOS.
* The **two-phase (separation train) variant uses Ideal + Henry's law**
  (H2O Raoult/Antoine; NH3 Henry Kpx; H2/N2 treated as non-condensable
  vapor-only species). Two IDAES limitations force this: eNRTL requires
  every species to be aqueous-phase-valid (incompatible with vapor-only
  H2/N2), and its activity coefficients are not wired into VLE fugacity
  anyway.
* Formation enthalpies (NBS aqueous standard states) are carried on every
  species, so reaction heat emerges from the energy balances without
  declared heats of reaction.

**To swap the thermo basis**: write another module exposing the same
`get_prop(phases)` signature and change one import line in
`ammonia_flowsheet.py`.

## Environment notes

* `idaes get-extensions` has no Fedora-aarch64 binaries; IPOPT 3.14.19 was
  installed into the `idaes-pse` conda env from conda-forge instead.
* Solve time: ~2 s (IPOPT), ~137 iterations from a cold start including
  three sequential passes around the gas recycle.
