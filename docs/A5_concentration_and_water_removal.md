# A5 electrolyte model, HCl integration and staged water removal

Run: `python idaes_model/water_removal.py` (`--conversion`, `--t-evap`). Tests: `test_brine_electrolyte.py`.

## A5: what was added
`idaes_model/brine_electrolyte.py` is a Pitzer ion-interaction model built from published parameters. Aspen's APV140 ELECNRTL set is licensed, so it could not be used. The model covers Na⁺, NH₄⁺, H⁺, Cl⁻, NO₃⁻, SO₄²⁻, HSO₄⁻ and OH⁻ and returns:
- osmotic coefficient, water activity and osmotic pressure
- ion activity coefficients
- boiling-point elevation (BPE)
- NaCl saturation index
- NH₄⁺/NH₃ and HSO₄⁻/SO₄²⁻ speciation
- the acid or base dose needed to reach a target pH

Against published data, NaCl φ and γ± agree within 0.005 from 0.1 to 6 mol/kg, and HCl and NH₄Cl within 0.01. The BPE of saturated NaCl is 9.1 K, compared with about 8.8 K in the literature.

Not covered: excess enthalpy. The heats of dilution behind the H1/C1 duty gap in A1 are still missing. Binary parameters other than NaCl are 25 °C values, and the model has no E-θ or ψ mixing terms.

The feed analysis is short of anions by 23 mol/h (0.16%). Na⁺ is adjusted to close the charge balance before any dosing is computed.

## A4: HCl stream
`ni2p_pfr.py` now takes `ph_target` (default pH 2) and reports the HCl dose in two parts:
- **pH setting:** the acid needed to bring the neutral feed to pH 2. Free H⁺ and HSO₄⁻ buffering both count; at 60 °C the HSO₄⁻ pKa is 2.4.
- **Reaction:** (1+S) protons consumed per mol of NO₃⁻ converted.

| Case | pH setting | Reaction | Total HCl |
|---|---|---|---|
| Default PFR (1.25% conversion) | 3.9 kg/h | 1.1 kg/h | 5.0 kg/h |
| Full conversion | ~4 kg/h | ~90 kg/h | ~94 kg/h |

Full conversion also needs about 54–58 kg/h NaOH to raise the pH to 11 before stripping. The acid and base together form about 2.3 kmol/h of extra NaCl, which counts against the saturation limit.

## Problem 1: feed concentration
The feed is 7,621 kg/h of water at 10.35 wt% TDS (I = 1.91 mol/kg, a_w = 0.937). Its osmotic pressure is 89 bar at 25 °C. At full conversion, the NH₃ leaves at only 0.27 wt% in that water.

Each way of concentrating the feed hits a limit:
- **RO:** with 120 bar elements, 1.1 polarization and a 13 bar margin, RO can remove only about 570 kg/h (7%). On the acidified effluent it removes essentially nothing.
- **NaCl saturation:** NaCl precipitates below about 2,100–2,200 kg/h of water. After allowing for the NaOH dose, the practical floor is about 2,500 kg/h (SI 0.93).
- **Crystallizer:** going below that floor needs a crystallizer that drops about 400 kg/h of solid NaCl.
- **H₂ solubility:** concentrating before the reactor (upstream) cuts equilibrium H₂ solubility to 0.57–0.63 of the raw feed through salting-out. The reactor is H₂-transfer-limited (A4), so this lowers conversion directly.

## Problem 2: staged water removal against stripping duty
All cases are at full conversion, with a 70 °C MVC evaporator and the 0.08 bar vacuum stripper, heat-integrated.

| Route | Water to stripper | Removal | Stripper heat |
|---|---|---|---|
| None | 7,621 kg/h | 0 | 313 kWth |
| Upstream RO + MVC | 2,487 kg/h | 95 kWe | 111 kWth |
| Downstream MVC (acidic, pH 2) | 2,501 kg/h | 93 kWe | 111 kWth |
| Downstream MVC + crystallizer | 1,250 kg/h | 138 kWe | 60 kWth |

- Each m³ of water removed saves **39 kWh_th** of stripper heat (the reboiler falls from 214 to 76 kW).
- MVC costs **18 kWh_e/m³** (BPE-aware compressor lift). RO costs 25 kWh_e/m³ for very little water.
- Whether removal pays depends on what the stripper heat costs:
  - **Heat-pump heat (COP ≈ 5):** the saving is worth only ~7.6 kWh_e/m³, so removal is a net loss (114 vs 61 kWe-eq).
  - **Electric or steam heat (COP ≈ 1):** removal wins (39 > 18).
  - **Free 60 °C reactor heat:** removal is not worth doing.
  - The break-even MVC energy is about 7.6 kWh_e/m³ with heat-pump heat.
- If water is removed, do it **downstream on the acidic effluent**:
  - The energy is the same as upstream.
  - The reactor keeps its H₂ solubility.
  - NH₃ lost to the condensate is only ~0.2 g/h.
  - The 5.1 t/h of clean condensate can supply electrolyzer make-up water (~90 kg/h needed).
- Concentrating upstream does lower the pH-setting acid, because there is less water to acidify.
- At the current Ni₂P conversion (1.25%) the energies are almost the same. Stripper duty depends on water flow, not on the NH₃ rate.

## Open items
- Excess enthalpy for A5.
- Cost of acid and base.
- Whether NaCl crystallizes in the stripper after the NaOH dose (SI 1.36 in the crystallizer case).
- An MVR stripper (S3) as an alternative to evaporating the water first.
- None of this is an IDAES flowsheet solve.
