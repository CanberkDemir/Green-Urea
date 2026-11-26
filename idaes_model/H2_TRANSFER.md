# Hydrogen delivery and nitrate hydrogenation screening model

`h2_transfer_reactor.py` is a **separate physical screening model** for the
aqueous nitrate reactor. It does not use the Aspen reaction rates or its
liquid-only H2 feed. It accounts for gas H2, dissolved H2, a liquid-to-catalyst
film, and charge-balanced nitrate/nitrite product pathways. No kinetic
prefactors or NH3 selectivity have been fitted to this brine or catalyst.
The intended replacement catalyst is now Ni₂P; see
`docs/A4_Ni2P_PFR_Parameters.md`. The optional reaction exponents below came
from Pd–Cu and must not be used as Ni₂P kinetics.

## Run the transport calculation

From the repository root:

```bash
python idaes_model/h2_transfer_reactor.py
python idaes_model/h2_transfer_reactor.py --kla-s 0.0002
python idaes_model/h2_transfer_reactor.py --k-no3 0.01 --k-no2 0.01 --kla-s 0.001
python idaes_model/h2_transfer_reactor.py --k-no3 0.01 --k-no2 0.01 --csv h2_profile.csv
python -m unittest idaes_model.test_h2_transfer_reactor
```

Defaults use the **unconcentrated** 8500 kg/h feed: 7621 kg/h water,
453.9 kg/h chloride, and 78.625 kg/h nitrate. Chloride gives a 1.68 mol/kg
NaCl-equivalent salinity. The 27 bar, 60 °C reactor geometry is carried over
for a size comparison, with **assumed** 50% liquid holdup, 1030 kg/m3 brine
density, gas-to-liquid `kLa = 0.001 s^-1`, and liquid-to-catalyst
`ks a = 0.01 s^-1`. These coefficients must be measured or estimated for the
actual contactor. Their units are per liquid volume. The script exposes them
through `--kla-s` and `--ksa-s`.

The default transport-only result is **12.0 mol H2/m3 brine** at equilibrium
(24.2 mg/L), or **0.20 kg/h dissolved H2** carried by the liquid at its
outlet. The remaining **9.80 kg/h is gas**. These are equilibrium/flow
estimates with no reaction; only a small dissolved inventory is needed if
gas continuously replenishes H2 consumed at the catalyst.

## Equations and numerical boundaries

The H2 equilibrium concentration is

`C* = Hcp(298 K) exp[500 K (1/T - 1/298.15 K)] p_H2 10^(-k_s m_NaCl)`.

`Hcp(298 K) = 7.8e-6 mol/(m3 Pa)` and the 500 K temperature coefficient
come from [Sander's 2023 H2 compilation](https://henry.mpch-mainz.gwdg.de/henry/casrn/1333-74-0).
The NaCl Sechenov form represents salting-out observed in
[Torín-Ollarves and Trusler (2021)](https://www.sciencedirect.com/science/article/pii/S037838122100087X).
**`k_s = 0.10 kg/mol` is a scenario value**, not a fitted mixed-brine
coefficient. Run `--sechenov 0` and `--sechenov 0.2` to bracket its effect.
At 27 bar, 60 °C, and the default salt proxy, the calculation is a screening
estimate; high-pressure fugacity, non-NaCl salts, and water-vapor pressure
are not included.

Along the reactor, gas-to-liquid transfer is `kLa(C* - C_liquid)` in
mol/(m3 liquid s). At the catalyst, quasi-steady H2 delivery obeys
`ks a (C_liquid - C_surface) = r_H2`. The local apparent reaction rates use
`C_surface` and conserve H2 and nitrogen. The outlet gas and liquid streams
are separate. Pressure and gas composition are assumed constant while gas
H2 remains; the gas balance still subtracts transferred H2.

The optional reaction network is balanced, with nitrate first reduced to
nitrite. Nitrite N then splits between NH3 and N2. The H2 demand is
`r_H2 = r_NO3 + [3 f_NH3 + 1.5 (1-f_NH3)] r_NO2`.
The apparent concentration exponents `0.2` for nitrate, `1.0` for nitrite,
and `0.2`/`0.3` for H2 follow one Pd-Cu/carbon study at 333 K:
[Mikami et al. (2003)](https://www.sciencedirect.com/science/article/pii/S0926337303000213).
The code expresses its H2 pressure exponents as surface-concentration
exponents only as a **scenario approximation**. `--k-no3` and `--k-no2` are
the rate at reference concentrations of 100 mol/m3 nitrogen ion and
10 mol/m3 surface H2, in mol N/(m3 liquid s). Both default to **zero**.
`--ammonia-branch` defaults to 0.5 for scenario runs; it is not measured.
No NH3 yield should be reported as a prediction until these are fitted.

For illustration only, with both reference rates set to 0.01 and the
default 50% NH3 branch, lowering `kLa` from 0.01 to 0.0002 s^-1 reduces
H2 reacted from about **7.1 to 1.0 kg/h**. This demonstrates why a
gas-liquid transfer coefficient matters even when gas H2 is plentiful.
These values are **not** catalyst performance estimates.
The optional CSV records sampled axial gas H2 flow, liquid and catalyst
surface H2 concentrations, nitrogen species, gas-liquid transfer rate, and
reaction rates. It lets the transfer and reaction zones be inspected directly.

The old B6 concentrate (`--water-kgph 1400`) is rejected: its chloride alone
is about 9.1 mol/kg water as NaCl-equivalent, beyond a credible single-phase
NaCl brine. Resolve salt precipitation or redesign B6 before using a
single-phase H2 solubility calculation for that stream.

## What to measure next

1. H2 equilibrium solubility in the actual mixed brine at 60 °C and the
   intended H2 partial pressure, or validate a mixed-salt thermodynamic model.
2. Gas-to-liquid `kLa`, liquid-to-catalyst `ks a`, liquid holdup, catalyst
   loading, and pressure drop for the proposed reactor geometry.
3. Nitrate, nitrite, NH3/ammonium, N2/N2O, and H2 inlet/outlet data over
   multiple H2 pressures and residence times. Fit rates and branching to
   those data, including pH and catalyst deactivation if observed.

The fixed-bed measurements of
[Batista et al. (1999)](https://www.sciencedirect.com/science/article/abs/pii/S0920586199001017)
show why this matters: in their reactor, H2 transfer from gas to liquid
dominated the observed nitrate disappearance rate.
