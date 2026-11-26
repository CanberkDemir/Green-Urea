# A4 — Ni₂P Catalytic Nitrate Hydrogenation PFR

**Project:** Green Urea  
**Status:** Provisional engineering design and kinetic screening model; not validated for industrial brine.  
**Purpose:** Replace the previous Pd–Cu catalyst basis with an ammonia-selective nickel phosphide candidate while retaining the continuous hydrogenation reactor architecture.

## 1. Evidence classification

- **Reported:** Experimental or material information attributed to the cited literature in the preceding research discussion.
- **Calculated:** Derived from reported values or explicitly stated assumptions.
- **Assumed:** Proposed engineering input for preliminary simulation, not a measured property.
- **Unknown:** Requires additional literature, supplier specifications or experiments.

The literature values below are compiled from the preceding research discussion; the primary manuscript and supporting information have not been independently re-extracted for this Markdown document. The supporting information was not fully examined in the preceding research. Verify reported values against the original experiment before final calibration.

## 2. Catalyst specifications

| Parameter | Symbol | Value | Basis |
|---|---|---:|---|
| Active phase | — | Ni₂P (dinickel phosphide) | [1] |
| Experimental support | — | Unsupported nanocrystals | [1] |
| Ni:P molar ratio | — | 2:1 | Stoichiometry |
| Nickel mass fraction | w_Ni | 79.12% | Calculated |
| Phosphorus mass fraction | w_P | 20.88% | Calculated |
| Formula molar mass | M | 148.36 g/mol | Calculated |
| Crystal structure | — | Hexagonal | [1] |
| Space group | — | P-62m | [1] |
| Activated crystallite diameter | d_c | 18 nm | [1] |
| Crystallite diameter after four cycles | — | 14 nm | [1] |
| Activation temperature | T_act | 400 °C | [1] |
| Activation duration | t_act | 1 h | [1] |
| Lattice parameter a = b | — | 5.872 Å | DFT value attributed to [1] |
| Lattice parameter c | — | 3.369 Å | DFT value attributed to [1] |
| BET surface area of experimental catalyst | S_BET | Unknown | Not verified |
| Intrinsic pellet porosity and tortuosity | — | Unknown | Not measured for proposed pellets |

**Important:** Crystallite size is not pellet diameter. The 18 nm value must not be used as the particle diameter in the Ergun equation.

## 3. Experimental reference conditions

| Parameter | Value | Basis |
|---|---:|---|
| Reactor type | Stirred gas–liquid–solid batch reactor | [1] |
| Vessel | Three-neck round-bottom flask | [1] |
| Liquid volume | 30 mL | [1] |
| Catalyst mass | 10 mg | [1] |
| Active catalyst concentration | 0.333 g/L | Calculated |
| Temperature | 60 °C (333.15 K) | [1] |
| Total pressure | Approximately 1 atm | [1] |
| Hydrogen concentration in gas | 5–10 vol% in Ar | [1] |
| Nominal H₂ partial pressure | 0.05–0.10 atm | Calculated from gas composition |
| Total gas flow | Approximately 5 mL/min | [1] |
| Agitation speed | 700–1,200 rpm | [1] |
| Initial nitrate concentration | 1.6–2.0 mmol/L | [1] |
| Initial pH for kinetic screening | 2 | [1] |
| Acidification | HCl or H₂SO₄ | [1] |
| Reaction duration | Typically 4–24 h | [1] |
| Representative high conversion | 99% | [1] |
| Representative high ammonia selectivity | 96% | [1] |

These are laboratory conditions, not industrial reactor specifications. The reported gas composition and the precise conditions associated with each kinetic and selectivity measurement should be checked in [1,2]. Atmospheric total pressure is not equivalent to 1 atm hydrogen partial pressure.

## 4. Proposed packed-bed PFR design basis

| Parameter | Symbol | Initial value | Evidence / status |
|---|---|---:|---|
| Active phase | — | Ni₂P | [1] |
| Proposed support | — | SiO₂ | Related hydrotreating literature [3]; nitrate activity unverified |
| Active phase loading | w_cat | 0.122 kg/kg supported catalyst | Related literature [3], provisional |
| Ni loading | — | 9.65 wt% | Calculated |
| P loading | — | 2.55 wt% | Calculated |
| Reactor type | — | Gas–liquid–solid packed-bed PFR | Assumed |
| Reactor length | L | 10 m | Previous project geometry, provisional |
| Internal diameter | D | 1 m | Previous project geometry, provisional |
| Pellet diameter | d_p | 2 mm | Assumed |
| Bed void fraction | ε_b | 0.40 | Assumed |
| Pellet porosity | ε_p | 0.50 | Assumed |
| Pellet tortuosity | τ_p | 3 | Assumed |
| Packed bulk density | ρ_b | 600 kg/m³ | Assumed; not a Ni₂P measurement |
| Temperature | T | 333.15 K | [1] reference |
| Reference total pressure | P | 1.013 bar absolute | [1] reference; not optimized |
| Alternative pressure | — | 27 bar | Unverified project scenario; not a literature operating condition |
| Initial pH for calibration | — | 2 | [1] reference |
| Controlled pH scenario | — | 3 | Assumed operating scenario, not calibrated kinetics |
| Thermal model | — | Isothermal | Assumed |
| Feed | — | Pretreated unconcentrated brine | Proposed |
| Effective activity factor | η_eff | Unknown | Sensitivity parameter; no validated default |

A silica-supported Ni₂P catalyst has been reported in hydrotreating research [3], but the 12.2 wt% loading and support specifications must be confirmed from that source before procurement. Neither nitrate selectivity nor nanoparticle activity can be assumed to transfer to supported pellets.

### 4.1. Geometry and catalyst inventory

For a fully packed cylindrical bed:

$$V_R=\frac{\pi D^2L}{4},\qquad W_{\rm supported}=\rho_b V_R,\qquad W_{\rm active}=w_{\rm cat}W_{\rm supported}.$$

| Calculated quantity | Value |
|---|---:|
| Cross-sectional area | 0.7854 m² |
| Packed bed volume | 7.854 m³ |
| Total supported catalyst mass | 4,712 kg |
| Active Ni₂P mass | 575 kg |
| Support mass | 4,137 kg |
| Nickel mass | 455 kg |
| Phosphorus mass | 120 kg |
| Empty-bed contact time at 8.625 m³/h | 54.6 min |
| Nominal void-volume residence time at ε_b = 0.40 | 21.9 min |

The latter assumes all bed voids contain liquid; actual liquid holdup in a three-phase bed must be determined separately. These inventories are geometrically calculated, not optimized or experimentally validated.

## 5. Keldgate feedstock design basis

From the project feedstock report [4]:

| Parameter | Value | Basis |
|---|---:|---|
| Average brine flow | 207 m³/day (8.625 m³/h) | Estimated Keldgate design basis |
| Peak brine flow | 450 m³/day | Estimated |
| Nitrate loading | 1,914 kg/day | Design mass balance |
| Nitrate concentration | 9.25 g/L (149 mmol/L) | Calculated |
| Chloride | 52 g/L | Cambridge Water analogue |
| Sodium | 44 g/L | Cambridge Water analogue |
| Sulphate | 1.8 g/L | Cambridge Water analogue |
| Phosphate as P | 0.29 g/L | Cambridge Water analogue |
| TOC | 148 mg C/L | Cambridge Water analogue |
| Untreated feed pH | 7–8 (reference 7.64) | Cambridge Water analogue |
| Theoretical NH₃ capacity | ~525 kg/day | Complete nitrogen recovery |
| Theoretical urea capacity | ~927 kg/day | Complete downstream conversion |
| H₂ demand for complete NH₃ pathway | ~249 kg/day | Stoichiometry with precise molar masses; [4] rounds to 247 kg/day |

The project brine nitrate concentration is approximately 75–93 times higher than the experimental concentration in [1]. Actual Keldgate co-contaminant concentrations have not been established. The previously considered concentrated stream must not be substituted without an electrolyte mass balance and salt precipitation assessment.

## 6. Preliminary kinetic model

### 6.1. Reference nitrate disappearance data

The following approximate points were extracted in the preceding research discussion from [1]. Confirm them from the original figure before formal regression.

| Time (h) | Nitrate (mmol/L) | Nitrate conversion |
|---:|---:|---:|
| 0 | 1.60 | 0% |
| 1 | 0.55 | 65.6% |
| 4 | 0.02 | 98.75% |

Assume pseudo-first-order nitrate disappearance at fixed experimental conditions:

$$-\frac{dC}{dt}=k_{\rm app}C,$$

$$k_{\rm app}=\frac{\ln(C_0/C_t)}{t}.$$

The 1 h and 4 h points give 1.068 h⁻¹ and 1.096 h⁻¹, respectively. Adopt a rounded screening estimate of 1.09 h⁻¹ (3.03 × 10⁻⁴ s⁻¹).

For the batch catalyst concentration of 0.3333 kg/m³:

$$k_W=k_{\rm app}\frac{V_L}{W}=9.08\times10^{-4}\ \mathrm{m^3\,kg_{Ni_2P}^{-1}\,s^{-1}}.$$

| Kinetic parameter | Initial value | Status |
|---|---:|---|
| Apparent batch rate constant | 1.09 h⁻¹ | Estimated from [1] |
| Active-mass-based rate coefficient k_W | 9.08 × 10⁻⁴ m³/(kg Ni₂P·s) | Calculated |
| Nitrate order | 1 | Assumed approximation |
| Reference ammonia selectivity S | 0.96 | [1], not established for brine |
| H₂ reaction order | Unknown | Requires data |
| Activation energy | Unknown | Requires temperature-dependent data |
| Arrhenius pre-exponential factor | Unknown | Requires data |
| Adsorption constants | Unknown | Requires mechanistic measurements |
| pH dependence | Unknown quantitatively | Requires fitting |
| Deactivation parameters | Unknown | Requires lifetime tests |

This is an apparent, condition-specific coefficient, not a published intrinsic kinetic constant. It incorporates the laboratory hydrogen availability, pH history, catalyst morphology and possible transport effects.

### 6.2. Lumped reaction network

Desired reaction:

$$\mathrm{NO_3^-+4H_2+2H^+\rightarrow NH_4^++3H_2O}.$$

Competing reaction (screening lump):

$$\mathrm{2NO_3^-+5H_2+2H^+\rightarrow N_2+6H_2O}.$$

Define reaction-extent rates per kg of active Ni₂P:

$$r_1=S k_W C_{NO_3^-},\qquad r_2=\frac{1-S}{2} k_W C_{NO_3^-}.$$

| Parameter | Value |
|---|---:|
| k₁ | 8.72 × 10⁻⁴ m³/(kg Ni₂P·s) |
| k₂ | 1.82 × 10⁻⁵ m³/(kg Ni₂P·s) |

The factor of 1/2 is necessary because reaction 2 consumes two nitrate ions per mole of reaction extent. Assigning all non-ammonia nitrogen to N₂ is a simplifying assumption; nitrite and other products must be included if measured. Concentrations are in mol/m³, and reaction rates are in mol/(kg Ni₂P·s).

### 6.3. Packed-bed material balances

With W denoting active catalyst mass and F_i in mol/s:

$$\begin{aligned}
\frac{dF_{NO_3^-}}{dW}&=-r_1-2r_2,\\
\frac{dF_{NH_4^+}}{dW}&=r_1,\\
\frac{dF_{N_2}}{dW}&=r_2,\\
\frac{dF_{H_2}}{dW}&=-4r_1-5r_2,\\
\frac{dF_{H^+}}{dW}&=-2r_1-2r_2,\\
\frac{dF_{H_2O}}{dW}&=3r_1+6r_2.
\end{aligned}$$

For axial position z, cross-sectional area A_c, and an effective activity correction η_eff:

$$\frac{dF_i}{dz}=\eta_{\rm eff}w_{\rm cat}\rho_b A_c\sum_j\nu_{ij}r_j.$$

Do not multiply by the support loading a second time when integrating on active catalyst mass. The proton balance is a stoichiometric source term, not a standalone pH model: electrolyte equilibria, buffering and acid addition must be included.

### 6.4. Idealized screening conversion

For constant liquid volumetric flow Q_L (m³/s), constant activity and nonlimiting hydrogen:

$$Da=\frac{\eta_{\rm eff}k_W W_{\rm active}}{Q_L},\qquad X=1-\exp(-Da),\qquad Y_{NH_3}=SX.$$

This model is intended for sensitivity analysis only. It cannot predict hydrogen starvation, pH dependence, adsorption saturation, nitrite accumulation, or catalyst aging.

### 6.5. Gas–liquid transfer and hydraulics

A more rigorous model should separate gas and dissolved hydrogen:

$$R_{H_2,GL}=k_La(C^*_{H_2}-C_{H_2,L}).$$

Here k_La is in s⁻¹ and R is in mol/(m³·s) on a consistently defined volume basis. Obtain C* using a temperature- and salinity-dependent Henry-law or electrolyte equilibrium model and hydrogen fugacity.

A single-phase Ergun baseline is:

$$-\frac{dP}{dz}=\frac{150\mu(1-\varepsilon_b)^2}{\varepsilon_b^3d_p^2}u_L+\frac{1.75\rho_L(1-\varepsilon_b)}{\varepsilon_b^3d_p}u_L^2.$$

This is not a validated three-phase pressure-drop correlation. Actual operation requires liquid holdup, catalyst wetting, gas velocity, effective diffusivities and multiphase pressure-drop treatment.

### 6.6. Runnable provisional PFR

`idaes_model/ni2p_pfr.py` implements the single-bed geometry in Section 4,
the batch-derived `k_W` and 96% reference selectivity in Section 6, and
separate cocurrent gas and liquid H₂ balances. It calculates local H₂
partial pressure from the H₂/N₂ gas composition, applies Henry/Sechenov
solubility, transfers H₂ into liquid with `kLa`, and balances a second
liquid-to-catalyst film flux against the lumped reaction demand. The H₂
saturation form is an assumed scenario normalized at the laboratory's
nominal 0.075 atm H₂ partial pressure; its half-saturation concentration
defaults to the corresponding pure-water equilibrium concentration.

```bash
python idaes_model/ni2p_pfr.py
python idaes_model/ni2p_pfr.py --kla-s 0.01
python idaes_model/ni2p_pfr.py --tubes 15
python -m unittest idaes_model.test_ni2p_pfr
```

At the provisional 27 bar, 10 kg/h H₂, single-bed and `kLa=0.001 s⁻¹`
case, the model gives 1.07% nitrate conversion, 0.222 kg/h NH₃ and
0.108 kg/h reacted H₂. At `kLa=0.01 s⁻¹` it gives 6.29% and 1.305 kg/h
NH₃. These outputs are **sensitivity scenarios**, not supported Ni₂P
performance estimates: `kLa`, catalyst-film transfer, effective activity,
H₂ saturation, and brine selectivity have not been measured. The model
does not yet resolve nitrite, pH, catalyst deactivation, heat balance,
pressure drop or precipitation. It is separate from the legacy IDAES PFR.

## 7. pH control and chemical demand

The preceding research discussion reports strong pH dependence in [1]: approximately complete nitrate conversion at initial pH 2 and 4 after 24 h, versus approximately 19% at initial pH 7. Verify the figure and exact conditions before fitting a pH-dependent rate law.

At acidic pH, the product is predominantly ammonium, not free molecular ammonia. Downstream recovery requires appropriate NH₄⁺/NH₃ speciation and potentially alkalization.

At 1,914 kg/day nitrate, assuming complete conversion exclusively to ammonium:

$$n_{NO_3^-}=1914/62.005\approx30.87\ \mathrm{kmol/day},$$

$$n_{H^+}=2n_{NO_3^-}\approx61.74\ \mathrm{kmol/day}.$$

| Acid | Ideal pure-acid equivalent |
|---|---:|
| HCl | ~2,251 kg/day |
| H₂SO₄ | ~3,027 kg/day |

**Correction to preceding discussion:** The earlier 1,514 kg/day H₂SO₄ estimate was a factor of two too low. One mole of sulfuric acid supplies up to two proton equivalents, giving approximately one mole H₂SO₄ per mole nitrate for the ammonium pathway. Actual consumption depends on acid dissociation, feed alkalinity, pH target, selectivity and chemical recycle. These are stoichiometric equivalents, not a dosing prescription.

Include acid and downstream base costs, corrosion, salinity changes, phosphate inhibition and Ni/P leaching in the economic and environmental assessment.

## 8. Validation priorities and unresolved parameters

| Missing input | Required evidence |
|---|---|
| Actual supported catalyst activity and selectivity | Nitrate hydrogenation experiments using the chosen supported material |
| Catalyst pellet diameter, density, porosity, tortuosity and BET area | Supplier datasheet or characterization |
| Nitrate concentration dependence | Measurements across the actual brine concentration range |
| H₂ dependence and pressure effects | Experiments at multiple hydrogen partial pressures |
| Activation energy | Multiple-temperature kinetic measurements |
| pH-dependent activity and selectivity | Controlled-pH experiments |
| Chloride, sulphate and phosphate effects | Synthetic brine and actual wastewater tests |
| Nitrite, N₂ and other nitrogen products | Complete nitrogen balance and time profiles |
| Catalyst leaching and deactivation | Long-duration experiments and elemental analysis |
| Gas–liquid and intraparticle transfer | Hydrodynamic and diffusion measurements/correlations |
| Final reactor dimensions and catalyst inventory | Validated productivity, pressure drop and process optimization |

**Recommended workflow:** reproduce the published batch experiment mathematically; fit and validate kinetics using complete concentration profiles; characterize supported Ni₂P; validate against real brine; then optimize packed-bed geometry and regenerate the process surrogate data. Do not treat the screening model as a certified design or use its predictions as validated economic inputs.

## 9. References

1. Wei, L. et al. *Mild and Selective Hydrogenation of Nitrate to Ammonia in the Absence of Noble Metals.* ACS Catalysis **2020**, 10, 3618–3628. DOI: [10.1021/acscatal.9b05338](https://doi.org/10.1021/acscatal.9b05338). [Open manuscript (OSTI)](https://www.osti.gov/servlets/purl/1609161).
2. Wei et al. Supporting Information for reference 1. [10.1021/acscatal.9b05338.s001](https://doi.org/10.1021/acscatal.9b05338.s001). Not fully extracted in the preceding research.
3. *Bifunctional Nature of a SiO₂-Supported Ni₂P Catalyst for Hydrotreating: EXAFS and FTIR Studies.* [Publisher record](https://www.sciencedirect.com/science/article/abs/pii/S0021951706000601). Related catalyst preparation literature, not evidence of nitrate hydrogenation performance; confirm precise loading and bibliographic details before citing in a manuscript.
4. Green Urea project, *UK Nitrate-Rich Wastewater Feedstock to Green Urea Plant Design.docx*. Sections on wastewater composition, design basis, mass balance and equipment sizing. Underlying site reference: [Keldgate WTW case study (2009)](https://waterprojectsonline.com/wp-content/uploads/case_studies/2009/Keldgate-WTW-2009.pdf). Brine chemistry analogue: [Growth of microalgae using nitrate-rich brine wash from the water industry](https://pureadmin.uhi.ac.uk/ws/portalfiles/portal/13795123/Growth_of_microalgae_using_nitrate_rich_brine_wash_from_the_water_industry.pdf).
5. Green Urea project, *Lumped Kinetics for Urea.txt*. Previous lumped Pd–Cu reaction-network notes; example rate coefficients were illustrative and are superseded for Ni₂P screening.

