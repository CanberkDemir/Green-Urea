# A4 — Experimental parameter dossier for Ni₂P nitrate hydrogenation

**Green Urea | Consolidated evidence dossier: nine PDFs supplied in `papers(1).zip`, plus targeted literature follow-up**

## 1. Decision and evidence status

Ni₂P remains a candidate, not a validated industrial catalyst. The supplied papers are sufficient for calibrating a dilute, unsupported-catalyst batch benchmark and choosing a physically plausible supported catalyst analogue. They are not sufficient to validate supported Ni₂P in Keldgate brine. The additional four papers provide a gas–liquid transfer correlation, hydrogen solubility and diffusivity models, and a real-brine Ru/C benchmark; see Sections 7–11. This document supersedes the earlier screening parameter recommendations where they conflict with the supplied measurements.

Labels: **E** = experimental report; **D** = derived from report; **A** = engineering assumption/model choice; **U** = unknown. A property measured for a different catalyst or reaction is an analogue, not a measurement of A4.

## 2. Direct nitrate hydrogenation evidence [1,2]

| Parameter | Value | Status / locator |
|---|---|---|
| Active phase | Ni₂P, unsupported nanocrystals | E, [1] Fig. 1–2 |
| Ni:P atomic ratio | 2:1 nominal | D, formula |
| Ni / P mass fraction in stoichiometric phase | 79.12 / 20.88 wt% | D |
| Formula molar mass | 148.36 g/mol | D |
| Crystallite size before/after activation | 18 nm | E, [1] p. 3619 |
| After four cycles | 14 nm | E, [1] p. 3621 |
| Activation in research study | H₂ annealing at 400 °C, 1 h | E, [2] Fig. S1 |
| Catalyst mass | 10 mg | E, [1] Fig. 2, [2] Table S1 footnote |
| Liquid volume | 30 mL | E, [1] Fig. 2 |
| Catalyst concentration | 0.333 kg/m³ | D |
| Reactor | Stirred three-neck batch flask, continuous gas flow | E, [1] Experimental Section |
| Temperature | 60 °C (333.15 K) | E |
| Total pressure | 1 atm | E |
| Gas composition | 5–10% H₂ in Ar | E |
| Nominal inlet H₂ partial pressure | 0.05–0.10 atm (dry gas basis) | D; not measured dissolved concentration |
| Gas flow | ca. 5 mL/min (mixture) | E |
| Stirring | 700–1200 rpm | E |
| Liquid matrix | Deionized water, NaNO₃ or NaNO₂ | E |
| Acid | HCl or H₂SO₄ | E |
| Initial pH | 2, 4, 7; recycling around 3 | E |
| Gas–liquid kLa, dissolved H₂, wetting | Unknown | U |

The experimental section prints “dry catalyst (10 mL)”, inconsistent with figure captions and Table S1; use the repeatedly documented 10 mg. Gas pressure in the abstract is shorthand; the methods explicitly specify diluted H₂. No pure-H₂ 1 bar intrinsic calibration should be inferred.

### 2.1 Nitrate time series (Table S1, rows 2-1 to 2-4, pp. S5–S6)

Initial nitrate 1.6 mmol/L; common conditions as above. Values are transcribed as reported; rounding means computed conversions/selectivities can differ slightly.

| Time (h) | pH start → end (as printed per interval) | NO₃⁻ remaining (mM) | NH₄⁺ (mM) | Conversion (%) | NH₄⁺ selectivity (%) | Rg (10⁻⁴ mol/h/g) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2 | 1.6 | not reported | 0 | — | — |
| 1 | 2 → 2–3 | 0.6 | 0.4 | 63 | 40 | 31.4 |
| 2 | 2–3 → 4 | 0.2 | 0.5 | 88 | 36 | 20.4 |
| 3 | 4 → 5 | 0.1 | 0.8 | 94 | 53 | 15.0 |
| 4 | 5 → 5 | 0.02 | 1.5 | 99 | 96 | 11.9 |

Main text p. 3620 instead gives 0.55 mM / 66% at 1 h. Preserve both rather than silently reconciling. Table S1 prints TOF 47 for row 2-4; TON 18.9 / 4 h implies 4.725 h⁻¹, so do not use 47 as a verified value. Row 2-1 reports 12.3 h⁻¹, whereas abstract-style summaries of TOF around 6 h⁻¹ are not a universal maximum.

Nitrite was not detected in the nitrate experiment according to [1]; detection does not establish absence at the catalyst surface. N₂ and NO were not quantitatively closed in this time series. The apparent unaccounted N (initial nitrate minus remaining nitrate minus measured ammonium) is 0.6, 0.9, 0.7 and 0.08 mM at 1–4 h respectively, subject to rounding and sampling. Do not assign it all to irreversible N₂; it decreases with time, so such an assignment would violate a cumulative product balance. The dataset cannot uniquely identify a mechanistic reaction network.

### 2.2 Other nitrate and nitrite runs (Table S1)

| Row | Feed (mM) | Time h | pH initial → final | Remaining mM | NH₄⁺ mM | Conversion % | Selectivity % |
|---|---|---:|---|---:|---:|---:|---:|
| 1, unactivated | NO₃ 1.6 | 4 | 2→2 | 1.6 | none | 0 | — |
| 3 | NO₃ 2.0 | 4 | 2→6 | 0.02 | 1.9 | 99 | 96 |
| 4 | NO₃ 1.6 | 24 | 7→7 | 1.3 | nd | 16 | nd |
| 5, no catalyst | NO₃ 1.6 | 4 | 2→2 | 1.6 | nd | 0 | nd |
| 6, no gas flow | NO₃ 1.6 | 24 | 7→7 | 1.6 | nd | 0 | nd |
| 11 | NO₂ 0.5 | 4 | 2→2–3 | 0.01 | 0.19 | 98 | 39 |
| 12 | NO₂ 1.6 | 4 | 2→2–3 | 0 | 0.20 | 100 | 13 |
| 13, closed labeled batch | ¹⁵NO₂ 1.6 | 4 | 2→7 | 0.24 | 0.72 | 85 | 53 |

nd = not determined. Main text reports 19% neutral conversion, whereas Table S1 reports 16%; 1.3 mM rounded remaining concentration implies 18.75%. Use this as uncertainty, not as a precise kinetic datum. At initial pH 4, complete reduction within 24 h is reported in [1] Fig. 3; no exact endpoint/detection bound supplied here. Labeled nitrite headspace GC-MS detected N₂ isotopologues and NO ([2] Fig. S8), but not a quantified nitrogen balance. Direct nitrite feeding decreases ammonia selectivity markedly.

### 2.3 Early-time nitrate disappearance (Table S1 rows 22, Fig. S4)

Initial 0.32 mM, pH 2, 60 °C. Rg as printed; no product measurements.

| Time min | Remaining mM | Conversion % | Rg (10⁻⁴ mol/h/g) |
|---:|---:|---:|---:|
| 2 | 0.302 | 5.6 | 16.5 |
| 4 | 0.286 | 10.6 | 15.1 |
| 6 | 0.298 | 6.8 | 6.70 |
| 8 | 0.274 | 14.4 | 10.3 |
| 10 | 0.265 | 17.2 | 9.97 |
| 12 | 0.256 | 20 | 9.51 |

The nonmonotonic concentration at 6 min suggests scatter. Treat these as reported measurements, not an exact deterministic trajectory. The first two Rg values correspond to 4.58×10⁻⁴ and 4.19×10⁻⁴ mol/kg/s (multiply the tabulated number by 0.1/3600). The rate is much less than would be implied by blindly extrapolating the high-concentration first-order model to brine.

### 2.4 Recycling and deactivation (Table S1 rows 8–10; [1] Fig. 3b)

| Feed mM | Cycle times h | Conversions % | Cumulative TON |
|---:|---|---|---|
| 1.0 | 12,12,12,12 | 84,78,61,64 | 9.9,19.1,26.3,33.9 |
| 1.5 | 12,12,12,12 | 85,76,47,47 | 15.1,28.6,36.9,45.2 |
| 2.0 | 10,12,8 | 84,74,32 | 19.8,37.3,44.9 |

The 1.5 mM series is controlled near pH 3. Continuous acid addition was used; MES and PIPES buffers failed to maintain pH in the authors' tests. Crystallite size changes and possible etching/repassivation motivate a deactivation state, but industrial lifetime and leaching remain unknown. No quantitative Ni/P leachate data were located.

### 2.5 Co-contaminant experiments (Table S1 rows 18–21)

2.0 mM nitrate, 4 h, initial pH 2. Baseline row 3: 99% conversion, 96% selectivity. Added NiO: 64%, 89%; Ni(OH)₂: 37%, 97%; Na₃PO₄: 36%, 100%; Ni₃(PO₄)₂: 28%, 95%. Final pH differs between tests (6–7). These demonstrate sensitivity to added species, not a quantitative phosphate inhibition constant or independently controlled pH effect.

## 3. Screening kinetic calibration (derived, not intrinsic)

For constant batch volume and pseudo-first-order nitrate disappearance:

$$C(t)=C_0\exp(-k_{app}t),\qquad k_W=k_{app}/(W/V_L).$$

Fit the four Table S1 concentrations by least squares in log space with fixed C₀=1.6 mM: $k_{app}=\sum t_i\ln(C_0/C_i)/\sum t_i^2$ = 1.033 h⁻¹ (approximately), giving $k_W=8.61\times10^{-4}$ m³/(kg active Ni₂P s). This fit is conditional on the changing-pH batch experiment, not a constant-pH law. Main-text 1 h and 4 h endpoints independently give 1.068 and 1.096 h⁻¹; the earlier 9.08×10⁻⁴ m³/(kg s) estimate is therefore a reasonable alternate screening value, not a published intrinsic constant.

| Condition | Derived apparent k_app (h⁻¹) | Derived k_W (m³/kg/s) | Caveat |
|---|---:|---:|---|
| Initially pH 2 | 1.033 | 8.607e-04 | Log fit, changing pH |
| Initially pH 4 | not identifiable | not identifiable | Complete conversion without exact residual |
| pH 7 | 0.00727–0.00878 | 6.05e-6–7.32e-6 | 16–19% conversion at 24 h |
| pH 3, first 1.5 mM recycling run | 0.156 | 1.30e-4 | 85%/12 h gives 0.158; remaining concentration 0.23 mM gives 0.156 |
| Nitrite 0.5 mM, initial pH 2 | 0.978 | 8.15e-4 | Endpoint, different substrate |

Do not estimate an exact k from zero residual in the 1.6 mM nitrite run without detection limits. No nitrate-specific activation energy or hydrogen reaction order is measured in these papers. DFT barriers are not apparent Arrhenius activation energies; [2] p. S12 reports N-adatom combination barriers 1.08 eV (P-rich Ni₂P(100)) versus 0.41 eV (Pd(111)), useful mechanistically only.

Measured batch-average nitrate productivity at 4 h, 1.6 mM: 11.9×10⁻⁴ mol/h/g = 1.19 mol/h/kg active = 3.31×10⁻⁴ mol/s/kg active. At 96% selectivity the equivalent ammonia productivity is approximately 0.0195 kg NH₃/(kg active h). This is a dilute experiment, not a sizing guarantee. At 79.75 kg nitrate/h (Keldgate) and this average rate, full conversion would require about 1,081 kg active catalyst, or about 8.83 t supported catalyst at 12.2% loading, before any transport or activity correction. This is a productivity-based comparison, not an optimized inventory; it differs materially from the earlier 575 kg assumption.

## 4. Supported catalyst evidence (other reactions)

### 4.1 Xin et al. [3], Table 1, p. 130

All samples have Ni₂P as the crystalline phase by XRD; Ni measured by AAS, accessibility by CO-TPD. Equivalent Ni₂P mass fraction calculated assuming all measured Ni belongs to stoichiometric Ni₂P; not a measured phase mass fraction.

| Crystallite size nm | Measured Ni wt% | Ni accessibility % | Equivalent Ni₂P wt% (D) |
|---:|---:|---:|---:|
| 3 | 8.9 ± 0.2 | 4.7 ± 0.1 | 11.25 |
| 7 | 9.1 ± 0.1 | 3.8 ± 0.3 | 11.50 |
| 15 | 9.8 ± 0.3 | 2.4 ± 0.2 | 12.39 |
| 24 | 8.8 ± 0.2 | 2.8 ± 0.1 | 11.12 |
| 38 | 14.1 ± 0.1 | 1.9 ± 0.1 | 17.82 |

The 15 nm sample is a coherent analogue close to the 18 nm unsupported nitrate catalyst. Use 9.8 wt% Ni and 12.39 wt% equivalent active phase as a named candidate, not an actual A4 supplier spec. Its phenol hydrogenation TOF peaks at 0.0488 s⁻¹; the optimum is reaction-specific.

Table 3, 473 K, phenol in dodecane: H₂ orders at 1–3 MPa are 1.01, 0.70, 0.53, 0.51, 0.49 for 3,7,15,24,38 nm respectively; phenol orders −0.08,0.15,−0.01,0.03,0.01. At 0.1–0.5 MPa, 7–38 nm samples show H₂ orders around 1. These establish size/coverage dependence for phenol, **not** nitrate-specific orders. Table 5 for 15 nm: dimensionless standard-state adsorption constants K°phenol=19.3±0.2, K°H₂=(4.1±0.7)×10⁻³, K°H-Ph=0.92±0.04, hydrogenation constant 0.89±0.10 s⁻¹ at 473 K. Do not transfer these constants to aqueous brine.

### 4.2 Yang, Liang & Prins [4], Table 1 and Experimental Section

Silica support nominal 500 m²/g, 63–90 µm support grains, pore volume after drying 1 mL/g. Table 1 footnote gives measured support surface area 478 m²/g. These refer to support, not finished mm-scale pellets.

| Sample | Reported Ni loading wt% | Finished BET m²/g | Ni₂P size TEM / XRD nm | CO uptake µmol/g |
|---|---:|---:|---|---:|
| NTA-5 | 5 | 335 | 5 / — | 51 |
| A-5 | 5 | 381 | 4 / — | 78 |
| N-5 | 5 | 428 | — / 11 | 13 |
| N-13 | 13 | 379 | 15 / 13 | 24 |
| NP-13 | 13 | 362 | — / 30 | 47 |

Different preparation methods produce different accessibility and crystallite sizes. The reported Ni loadings are not necessarily finished active phase mass fractions. These hydrotreating catalysts are not tested in nitrate brine. Do not combine their surface area with Xin's loading and present it as one measured material.

### 4.3 Lan, Hensen & Weber [5], Table 1, p. 123 (precursors)

| Precursor | Ni wt% | P wt% | BET m²/g | Pore volume cm³/g | BJH pore diameter nm |
|---|---:|---:|---:|---:|---:|
| Ni(NO₃)₂–PO₄ | 5.0 | 5.3 | 90 | 0.35 | 11.7 |
| Ni(NO₃)₂–PO₃ | 5.6 | 6.0 | 212 | 0.90 | 15.5 |
| Ni(NO₃)₂–PO₂ | 6.3 | 6.8 | 180 | 0.75 | 12.8 |
| NiO–PO₄ | 6.3 | 6.5 | 129 | 0.57 | 13.3 |
| NiO–PO₃ | 6.9 | 6.9 | 159 | 0.70 | 13.0 |
| NiO–PO₂ | 6.9 | 7.0 | 159 | 0.72 | 12.8 |

These are precursor properties, not finished Ni₂P pellets. P/Ni ratios 1.9–2.0 are far above stoichiometric Ni₂P (0.5); excess phosphorus and other phases are relevant. Table 3 CO uptake spans 8–100 µmol/g for reduced samples; HDS TOFs 61–247 h⁻¹. Neither is a nitrate rate. Preparation changes reduced phase composition and accessibility; do not infer active fraction from total phosphorus alone.

## 5. A4 implementation and remaining engineering parameters

| Parameter | Value / treatment | Status |
|---|---|---|
| Catalyst choice | Ni₂P/SiO₂ candidate, 15 nm Xin analogue | A/E analogue |
| Active loading | 12.39 wt% equivalent from 9.8 wt% Ni; or retain 12.2% as old sensitivity case | D/A |
| Pellet diameter | 2 mm | A, not demonstrated by supplied studies |
| Pellet porosity / tortuosity | 0.50 / 3 | A |
| Bed void fraction / bulk density | 0.40 / 600 kg/m³ | A |
| Reactor dimensions | 10 m × 1 m diameter | A, existing design |
| Bed volume / catalyst inventory | 7.854 m³ / 4,712 kg | D |
| Active mass (12.39% equivalent) | 584 kg | D |
| Temperature | 333.15 K | E reference |
| Pressure | 1 atm total for lab validation; 27 bar only unverified case | E/A |
| Liquid flow | 8.625 m³/h unconcentrated | D, project report |
| Nominal void residence time | 21.9 min ignoring gas holdup | D/A |
| Inlet pH / controlled pH | 2 batch reference; pH 3 controlled screening | E/A |
| kLa, liquid-solid transfer, H₂ diffusivity, Henry coefficient | Unknown for brine | U |
| Nitrate kinetics, NH₄ selectivity | Calibrate batch first; supported/brine correction unidentified | D/U |

### 5.1 Mass balances and reaction network

A conservative lumped screening reaction is NO₃⁻ + 4H₂ + 2H⁺ → NH₄⁺ + 3H₂O. Other possible balanced overall/elementary lumped reactions include:

- NO₃⁻ + H₂ → NO₂⁻ + H₂O
- NO₂⁻ + 3H₂ + 2H⁺ → NH₄⁺ + 2H₂O
- 2NO₂⁻ + 3H₂ + 2H⁺ → N₂ + 4H₂O
- NO₂⁻ + 0.5H₂ + H⁺ → NO + H₂O

These are stoichiometric bookkeeping choices, not a proven elementary mechanism. A surface intermediate pathway may be necessary to reconcile low measured bulk nitrite and changing selectivity. Do not force the 13% nitrite-feed selectivity onto all nitrate-derived intermediates. Fit measured nitrate and ammonium together; treat unmeasured products as latent, enforce nitrogen conservation, avoid claiming unique parameters.

For rates per kg active Ni₂P and reactor coordinate z:

$$\frac{dF_i}{dz}=A_c\rho_b w_a\sum_j\nu_{ij}r_j.$$

For H₂, define gas–liquid transfer per bed volume explicitly:

$$R_{GL}=k_La(C^*_{H_2}-C_{H_2,L}),\quad C^*=H^{cp}(T,\text{salinity})f_{H_2}.$$

$$\frac{dF_{H_2,G}}{dz}=-A_cR_{GL},\qquad \frac{dF_{H_2,L}}{dz}=A_cR_{GL}-A_c\rho_b w_a\sum_j b_jr_j.$$

Here b_j is positive H₂ consumption per reaction extent; kLa is based on bed volume. If a correlation uses liquid-volume basis multiply by liquid holdup. Hcp is concentration/fugacity form; do not invert convention inadvertently. Add liquid-solid transfer and pellet diffusion as separate resistances; $D_{eff}=\varepsilon_pD/\tau_p$ is a provisional porous-medium closure. Use electrolyte activities for pH, not independent free-H⁺ flow alone. The original five catalyst papers do not calibrate these transport closures; the additional papers supply correlation-based estimates, not A4-specific validation. Avoid double counting apparent batch transport limitations in an intrinsic kinetic model.

## 6. Brine and plant-scale transferability

Project report *UK Nitrate-Rich Wastewater Feedstock to Green Urea Plant Design*, design basis: 207 m³/day, 1,914 kg/day nitrate, 9.25 g/L nitrate (~149 mM), 52 g/L chloride, 44 g/L sodium, 1.8 g/L sulphate, pH 7.64 analogue, phosphate as P 0.29 g/L, TOC 148 mg C/L. Major ions are Cambridge Water analogues, not measured Keldgate composition. Alkalinity and charge-balanced composition are unknown. Brine nitrate concentration is 75–93× the main experimental range. Do not extrapolate first-order law linearly over that range as a validated prediction. Concentration to 1,400 kg/h water requires a separate salt precipitation balance.

For complete conversion of 1,914 kg/day nitrate: ~30.87 kmol/day nitrate, ~247–249 kg/day H₂ (depending on molar mass convention), ~526 kg/day NH₃ equivalent. Complete conversion to ammonium consumes 61.74 kmol H⁺ equivalents/day, or ~2,251 kg/day pure HCl / ~3,028 kg/day pure H₂SO₄, excluding alkalinity and speciation effects. Subsequent conversion to free NH₃ requires base or an equivalent separation strategy. No actual acid usage or corrosion data are provided for A4.

## 7. Hydrogen solubility: replacement for the assumed Henry coefficient [6]

Torín-Ollarves and Trusler measured H₂ solubility in water and 2.5 mol/kg NaCl at 323.15–423.15 K, up to about 40 MPa (expanded relative uncertainty 3%, k=2). Their model also uses literature data down to 273.15 K and up to 101 MPa. It is a NaCl model, not a multicomponent Keldgate model.

### 7.1 Selected measured equilibrium data (Tables 3–4)

| T K | NaCl mol/kg water | Bubble pressure MPa | H₂ molality mol/kg water |
|---:|---:|---:|---:|
|323.15|0|15.80|0.1121|
|323.15|0|32.97|0.2264|
|323.15|2.5|12.01|0.0611|
|323.15|2.5|26.23|0.1231|
|323.15|2.5|39.47|0.1900|
|373.15|2.5|11.64|0.0638|
|373.15|2.5|22.94|0.1306|
|373.15|2.5|45.81|0.2365|
|423.15|2.5|14.94|0.0959|
|423.15|2.5|38.73|0.2541|

### 7.2 Implementable correlation (Eqs. 18–22, Table 5)

Define $\theta=T/273.15-1$. The extended Krichevsky–Kasarnovsky equation is

$$b_{H_2}=\frac{f_{H_2}}{H_b(T)}\exp[-k_s(T)b_s-\bar V_{H_2}(T)(P-P_{sat})/(RT)].$$

Use fugacity and Henry constant in the same pressure units; use Pa and m³/mol in the Poynting correction. The polynomial coefficients below are directly from Table 5. The numerical Henry constant is on a molality basis (MPa per mol/kg water).

| coefficient index | a (ln Henry) | c (partial molar volume, cm³/mol) | d (Sechenov, kg/mol) |
|---:|---:|---:|---:|
|0|4.6449|19.615|0.2898|
|1|3.3252|-7.6400|-1.4330|
|2|-10.901|33.425|3.9584|
|3|8.0526|—|-3.1666|

$H_b=\exp(\sum a_i\theta^i)$, $\bar V=10^{-6}\sum c_i\theta^i$, $k_s=\sum d_i\theta^i$. Convert molality to concentration with water mass per solution volume, $C^*=b_{H_2}\rho_L w_{water}$. Use local fugacity $f=\phi_{H_2}y_{H_2}P$ and water vapor correction; do not treat total pressure as hydrogen partial pressure in diluted streams.

```python
import math

def hydrogen_equilibrium(T, P_MPa, f_MPa, psat_MPa, salt_molality, water_kg_m3):
    theta = T/273.15 - 1
    H = math.exp(4.6449 + 3.3252*theta - 10.901*theta**2 + 8.0526*theta**3)
    v = 1e-6*(19.615 - 7.6400*theta + 33.425*theta**2)
    ks = 0.2898 - 1.4330*theta + 3.9584*theta**2 - 3.1666*theta**3
    b = f_MPa/H*math.exp(-ks*salt_molality-v*(P_MPa-psat_MPa)*1e6/(8.314462618*T))
    return b*water_kg_m3  # mol/m3 of solution
```

At 333.15 K: H_b=139.03 MPa/(mol/kg), partial molar volume=19.55 cm³/mol, k_s=0.13246 kg/mol. Example with f≈P_H2, total pressure approximated by P_H2 and water vapor pressure 0.01995 MPa: at 0.101325 MPa H₂ and 1.5 mol/kg NaCl, b≈0.000597 mol/kg; at 2.7 MPa b≈0.01562 mol/kg. With assumed 900 kg water/m³ solution, C*=0.537 and 14.06 mol/m³ respectively. These are illustrative ideal-fugacity calculations, not measured A4 properties; use consistent wet gas composition in production code.

The paper's fitted standard uncertainty is 1.2% for its selected database. However independent Chabab measurements at 3 mol/kg showed ~25% lower solubility; the alternative inferred Sechenov coefficient was ~0.22 kg/mol. Use this as a sensitivity case, not a statistical confidence interval. Keldgate's chloride reference 52 g/L is ~1.47 mol/L chloride, not necessarily 1.47 mol/kg NaCl: determine actual water content and ionic balance first. Concentrated brine outside model domain requires an electrolyte model or measurements.

## 8. Hydrogen diffusivity [7]

Piszko et al. measured transport properties by dynamic light scattering at 293–393 K and pressures up to 20 MPa. Their Eq. (14) includes gas mixture composition, but for H₂ only (methane fraction zero) reduces to

$$D_{H_2}=4.182\times10^{-11}\exp\left(\frac{1.380649\times10^{-23}T}{8.425\times10^{-22}}\right)(1-0.009703P_{MPa})\quad[m^2/s].$$

This is the authors' empirical expression (positive exponent proportional to T), not an Arrhenius rate law. It gives 9.83e-9 m²/s at 333.15 K and zero-pressure extrapolation; 9.57e-9 m²/s at 2.7 MPa. The average absolute relative deviation of the full Eq. 14 is 16%; mean expanded uncertainty across gas-diffusion datasets is 29%, and ~15% for binary water/H₂. Crucially, dissolved gas diffusivity in brine was accessible only at salt mass fractions up to 0.02 and T=293 K. Higher concentrations and T>300 K interfered with the scattering measurement. The correlation has no explicit salt dependence and does not validate D in hot concentrated Keldgate brine. Use a separate uncertain salinity factor, not salt diffusivity as H₂ diffusivity.

With assumed pellet porosity 0.50 and tortuosity 3, $D_{eff}=1.60\times10^{-9}$ m²/s at 2.7 MPa before salinity correction. These pellet properties remain unmeasured. For a spherical pellet and genuinely first-order local kinetics, $\phi=R_p\sqrt{k_v/D_{eff}}$, $\eta=3(\phi\coth\phi-1)/\phi^2$. Here k_v must be on pellet-volume basis and for the diffusing species in question. The nitrate-based apparent rate constant cannot be used as H₂ first-order k_v. For coupled diffusion solve the pellet reaction–diffusion equations for both H₂ and nitrate, and do not apply a second effectiveness correction to an already transport-limited rate without justification.

## 9. Gas–liquid mass transfer [8]

Iliuta et al. provide neural correlations for downflow cocurrent trickle beds: separate liquid-side kLa, gas-side kGa and gas–liquid area. Approximately 3,200 measurements; database pressure 0.1–5.1 MPa, temperature 19–30.5 °C, equivalent particle diameter 0.54–26.4 mm, column diameter 0.023–0.30 m, voidage 0.263–0.94. A4's 60 °C and 1 m column exceed those ranges. Pure H₂ is not among the gases listed in Table 1. It is a sensitivity correlation, not a validated prediction. Table 5 generalization AARE: liquid 22.9%, gas 13.1%, area 29.1%. Those errors apply to their database, not extrapolation to A4.

### 9.1 Liquid-side neural correlation (Table 3 Eqs. 1–5; Table 4, p. 5638)

Definitions: $Re_L=u_L\rho_Ld_p/\mu_L$, $We_L=u_L^2d_p\rho_L/\sigma_L$, $X_G=u_G\sqrt{\rho_G}/(u_L\sqrt{\rho_L})$, $Mo_L=g\mu_L^4/(\rho_L\sigma_L^3)$, $Sc_L=\mu_L/(D_L\rho_L)$. Superficial velocities use empty column area. The paper defines Krischer–Kast hydraulic diameter $d_h=d_p[16\epsilon^3/(9\pi(1-\epsilon)^2)]^{1/3}$; external packing-plus-wall area $a_s=6(1-\epsilon)/(\psi d_p)+4/D_c$ with sphericity ψ; $S_b=a_sd_h/(1-\epsilon)$; $Sh_L=k_La d_h^2/D_L$. Check the original notation (p. 5641) if implementing nonspherical pellets.

Normalize the six inputs by $U_i=\log_{10}(x_i/m_i)/s_i$ and add bias U7=1. Hidden layer sigmoid, output sigmoid; Sh_L=0.3359 × 10^(5.2428 S). Exact liquid network coefficients transcribed from Table 4:

```python
import numpy as np

def iliuta_kla(ReL, WeL, XG, MoL, ScL, Sb, dh, DL):
    x = np.array([ReL, WeL, XG, MoL, ScL, Sb])
    lo = np.array([1.994e-2, 1.851e-6, 2.636e-3, 2.545e-15, 44.33, 2.552])
    span = np.array([4.898,6.298,4.918,10.114,4.042,1.9011])
    U = np.r_[np.log10(x/lo)/span, 1.]
    W = np.array([
      [1.7352,-1.0859,4.4131,5.9993,5.5325,-0.8589,2.7195,1.8965,-0.4475,-1.7098,1.9772,-1.8640,6.8217],
      [3.7209,4.3527,-0.9842,2.2940,-1.2809,1.5063,-2.1694,5.1352,2.4182,-3.1443,-7.2618,6.3021,7.4821],
      [-3.1528,-5.9518,1.4652,-5.1562,-0.7164,-0.6652,-1.2966,4.2572,-1.2258,-0.5253,-4.1405,-2.4334,2.5210],
      [10.8069,0.3045,2.5158,4.6663,-3.2695,4.2675,-4.6230,-3.7013,0.6073,-0.7606,0.1795,2.5706,0.0143],
      [-5.7494,-0.8630,6.0466,3.2448,2.0916,-7.2309,0.4930,-1.9482,0.1645,4.6749,1.1673,2.0291,0.6759],
      [9.2444,1.2997,10.0231,8.7118,-5.8084,-5.6044,2.3693,11.4437,9.7706,-6.4817,3.1763,4.8084,8.9547],
      [-5.0819,5.1608,-6.9573,3.3028,3.9598,5.6439,2.6890,-5.2396,-1.6727,-2.3656,4.6354,-1.7145,1.2650]])
    v = np.array([5.2505,-3.9571,7.6703,4.3998,9.9828,-8.1581,6.3469,-4.3963,-7.0899,3.4881,-6.1047,-4.3143,5.5141,-4.4515])
    sigmoid = lambda z: 1/(1+np.exp(-np.clip(z,-700,700)))
    S = sigmoid(np.r_[sigmoid(U@W),1.]@v)
    Sh = 0.3359*10**(5.2428*S)
    return Sh*DL/dh**2, U[:-1] # s^-1, normalized inputs (flag any outside 0..1)
```

This code is a transcription, not a validated implementation against the authors' dataset. Verify against Table 4 and test before deploying. In particular the gas-side and area networks (Table 3 Eqs. 6–12 and Table 4) are separate, and are not required if liquid-side resistance dominates and you only need volumetric kLa. Do not multiply kLa by area a again. If gas-side resistance is significant, combine on a consistent driving-force basis; for concentration-based equilibrium C_L=m C_G, $1/K_La=1/k_La+m/k_Ga$. Need gas diffusivity, viscosity and local pressure/composition for gas-side model. The paper does not supply liquid–solid transfer, catalyst wetting, liquid holdup or pressure drop closures.

### 9.2 Engineering balance and hydrogen-transfer requirement

At average feed, nitrate load ≈1286 mol/h, stoichiometric H₂ consumption ≈5144 mol/h =1.43 mol/s at complete NH₄ production. At 7.854 m³ bed volume, mean H₂ uptake required ≈0.182 mol/(m³ bed s). Therefore a necessary optimistic lower bound is kLa ≥0.182/C* (assuming liquid concentration zero, uniform C*, no transport resistance beyond gas–liquid, no feed dissolved H₂). With Section 7 example at 1 bar H₂: ≥0.34 s⁻¹; at 27 bar: ≥0.013 s⁻¹. This explains why H₂ pressure can matter dramatically independently of intrinsic kinetics. Actual required kLa will be higher for nonzero liquid H₂ or incomplete wetting. These are demand bounds, not measured transfer coefficients.

For the PFR, use local $u_G=\dot n_G ZRT/(PA_c)$, $u_L=Q_L/A_c$. At 8.625 m³/h, uL=0.00305 m/s. Set gas recycle/purge from a gas balance, not arbitrary excess treated as dissolved H₂. Liquid–solid flux per bed volume is $k_{LS}a_w(C_L-C_s)$, with $a_w=f_w a_p$, $a_p=6(1-\epsilon_b)/(\psi d_p)$. For a preliminary fully wetted single-phase comparison only, a standard packed-particle closure is $Sh=2+1.1Re_p^{0.6}Sc^{1/3}$ (Wakao and Funazkri, 1978; [13]); it is not a high-pressure trickle-bed correlation. Actual wetting, holdup and transfer are unidentified. Use $\tau_L=\epsilon_L V/Q_L$ with $\epsilon_L$ liquid volume per bed volume, not total bed voidage unless flooded. Single-phase Ergun is not a multiphase pressure drop prediction. For pressure drop/holdup see [10–12].

## 10. Real ion-exchange brine benchmark — Huo et al. [9]

This is the strongest process-specific experimental benchmark in the supplied archive. **The catalyst is Ru/C, not Ni₂P.** It is an alternative candidate, not a source of Ni₂P rate parameters. The nitrate concentration is almost identical to the Keldgate design basis.

### 10.1 Catalyst and reactor

| Parameter | Experimental value / locator |
|---|---|
| Catalyst | Commercial 5 wt% Ru/C powder, Sigma-Aldrich (Section 2.1) |
| BET area | 860 m²/g (prior characterization cited by authors) |
| Ru dispersion | 38% (prior characterization) |
| Vessel | 250 mL three-neck stirred flask, synthetic feed 120 mL |
| Agitation | 1100 rpm |
| H₂ | 1 atm headspace maintained with ~300 mL/min flow |
| Catalyst concentration | 1–7.5 g/L tested, 5 g/L baseline synthetic, 6 g/L real-brine runs |
| Temperature | 25–35 °C, baseline 30 °C |
| Nitrate | 50–200 mM synthetic, 149 mM real |
| Reactor type | Stirred semi-batch, not packed-bed |
| CO₂ flow during buffered run | 65 mL/min |
| Catalyst recycling | Four uses; activity increased after first use; Ru detected <0.2% of initial Ru inventory, dissolution vs attrition unresolved |

### 10.2 Actual brine characterization (Table 1)

| Species | California utility brine [9] | Keldgate design/analogue |
|---|---:|---:|
| pH | 8.5 | 7–8 assumed |
| Nitrate mg/L | 9,260 | 9,250 |
| Chloride mg/L | 13,700 | 52,000 |
| Sodium mg/L | 16,900 | 44,000 |
| Sulphate mg/L | 4,350 | 1,800 |
| Alkalinity mg/L as CaCO₃ | 13,100 | Unknown |
| Inorganic C mg/L | 3,057 | Unknown |
| Organic C mg/L | 22.5 | 148 |
| Phosphate | BQL | 290 mg P/L analogue |
| Nitrite | BQL | Unknown |
| K, Mg, Ca mg/L | 35.7, 15.8, 5.8 | Unknown |
| Cr, V, Li, Mo, Sr mg/L | 2.9, 1.2, 0.6, 0.5, 0.2 | Unknown |

BQL: below quantification limit; instrument limit 0.1 mg/L and method limit 10 mg/L for dilution factor 100. Keldgate's chloride is substantially higher. The Huo study cannot validate salt tolerance of Ni₂P, and even Ru/C requires testing against the actual Keldgate matrix.

### 10.3 Kinetic measurements (Table 2, Section 3.1)

Reported apparent initial zero-order k0 values, with the authors' normalization convention (printed mM min⁻¹ gRu⁻¹). Their Eq. 2 multiplies k0 by Ru loading in g/L; the printed units and normalization are not dimensionally consistent as written. Preserve the table and use raw concentration-time profiles or verify normalization with authors before converting to mass-based rates. The values below are not fitted Ni₂P coefficients.

| NaCl wt% | HCO₃ mM | SO₄ mM | NO₃ mM | T °C | pH | k0 |
|---:|---:|---:|---:|---:|---|---:|
|5|100|100|100|30|Drift|0.30±0.03|
|5|100|100|50|30|Drift|0.70±0.07|
|5|100|100|200|30|Drift|0.08±0.01|
|0|0|0|100|30|7.5|0.09±0.01|
|0|0|0|100|30|11|0.11±0.01|
|0|0|0|100|30|Drift|0.15±0.01|
|5|0|100|100|30|Drift|0.16±0.03|
|0|100|0|100|30|Drift|0.58±0.01|
|10|100|200|100|30|Drift|0.10±0.01|
|5|100|100|100|25|Drift|0.18±0.01|
|5|100|100|100|35|Drift|0.32±0.04|

Apparent nitrate order −1.5 across 50–200 mM; apparent activation energy 45±6 kJ/mol. No H₂ order established in this study. In real brine k0=1.80±0.04 versus 1.24±0.02 for synthetic mimic. Bicarbonate enhanced activity while chloride and sulphate inhibited it for Ru/C. This is evidence that a simple positive first-order extrapolation is dangerous at high concentration, not evidence that Ni₂P has negative order.

### 10.4 Real brine performance (Fig. 5, Table 4, Sections 3.3–3.4)

At 30 °C and 1 atm H₂, CO₂ co-feed maintained pH about 7.6–7.9 in the small-scale test. After 8 h, 85% ammonia retained in liquid with CO₂ versus 51% without (pH reached 10 without buffer). At scale-up to 1 L reactor, after up to 24 h with CO₂, nitrate decreased from 149 to 1.2 mM; total ammonia reached 136 mM from <3.2 mM detection limit. This implies 99.2% nitrate conversion and approximately 90–92% liquid nitrogen recovery as ammonia, not a complete gas-phase selectivity measurement. The subsequent membrane step reduced total ammonia to 9 mM and captured ammonium sulphate; overall nitrogen recovery about 80%. Approximately 7% remained in treated brine and remainder presumed lost to gas during hydrogenation. The process is not a direct anhydrous-NH₃ product route.

| Component | Initial | After hydrogenation | After membrane |
|---|---:|---:|---:|
|pH|8.5|8.6|9.9 (NaOH adjustment)|
|Cl mg/L|13700|14200|14200|
|HCO₃/CO₃ mM|232|485|484|
|NO₃ mg/L|9260|79|75|
|SO₄ mg/L|4350|4520|4280|
|Total ammonia mM|<3.2|136|9|

Membrane: PTFE polypropylene-backed, nominal pore size 0.45 µm, three 195 cm² cells, feed and acid flow 1.5 L/min, real-brine feed 4 L, acid absorbent 1.15 L of 0.25 M H₂SO₄, 30 °C. Real-brine K_MD=0.11 m/h and initial flux 199 g NH₃-N/m²/h (single run). Synthetic mimic K_MD=0.12±0.01 m/h, flux 225±4 g N/m²/h. These are ammonia membrane transfer parameters, not hydrogen kLa.

## 11. Final parameter register and validation plan

| Item | Recommended model treatment | Evidence |
|---|---|---|
| Active phase | Ni₂P, candidate; compare Ru/C alternative | [1,9] |
| Supported loading | 12.39 wt% equivalent; not a purchased catalyst | [3], derived |
| Bed geometry | 10 m × 1 m; 2 mm pellets; voidage 0.4; bulk density 600 kg/m³ | A |
| Active mass | 584 kg on equivalent-loading basis | D |
| Nitrate disappearance | k_W=8.61e-4 m³/kg/s at lab benchmark only | [1,2], fitted |
| NH₄ selectivity | Fit time-dependent batch data, do not force fixed 96% | [2] |
| Nitrite kinetics | Endpoint constraints, separate experimental feed cases | [2] |
| H₂ reaction order, activation energy | Unknown for nitrate/Ni₂P; do not import phenol/Ru values | U |
| pH | Explicit equilibrium and proton balance; sensitivity, not validated kinetic pH function | [1,2] |
| H₂ equilibrium | Extended KK polynomial with NaCl approximation | [6] |
| H₂ liquid diffusivity | Eq. 14 pure-H₂ simplification, uncertain salinity correction | [7] |
| Gas–liquid kLa | Iliuta liquid network, extrapolative at A4 dimensions/T | [8] |
| Liquid–solid transfer | Fully wetted screening closure only; require high-pressure model | [10–13] |
| Pellet porosity/tortuosity | 0.50/3 provisional; effective diffusivity sensitivity | A |
| Liquid holdup / wetting / two-phase pressure drop | Unknown; need dedicated correlations or measurements | [10–12] |
| Brine concentration | 149 mM nitrate; actual co-ion composition/alkalinity unknown | Project report |
| Long-term activity, leaching | Unknown for supported Ni₂P in brine | U |

### 11.1 Suggested implementation order

1. Reproduce Table S1 nitrate and ammonium curves at the original batch conditions. Treat unaccounted nitrogen as a latent pool, not measured N₂. Check nitrogen and hydrogen balances, positivity and units. Fit on concentration data, not simultaneous duplicated conversion and selectivity values.
2. Compare candidate kinetic structures using profile likelihood or parameter sensitivity. A first-order apparent fit is a baseline, not intrinsic chemistry. A saturation/inhibition model with nitrate, dissolved H₂ and pH dependence may be more appropriate but parameters are unidentified. Use reference-normalized factors so the apparent calibration is preserved at its benchmark; avoid double counting mass transfer.
3. Add equilibrium solubility, gas-liquid transfer, pellet transfer, electrolyte speciation, acid/base dosing, liquid holdup, pressure profile and temperature balance as separate modules. Validate each separately. Use local gas and liquid velocity and concentrations; implement bounds and an explicit out-of-domain flag.
4. Evaluate ranges of effective activity, H₂ pressure, kLa, wetting and salinity, and report feasible production ranges rather than a single optimized result. Do not generate final optimization surrogate training data from unvalidated extrapolations without marking the resulting model as scenario-based.
5. Compare Ni₂P against Ru/C on ammonia yield, actual productivity, catalyst replacement, acid/base use, recovery duty, leaching, and total annualized cost. Ru/C has stronger direct feedstock evidence in this archive, but its industrial pellet scale-up is also unvalidated.

### 11.2 What additional papers are worth obtaining?

The nine papers are enough for a documented screening model but not for a predictive industrial model. A targeted web search found the following additional hydrodynamics sources; their full equations have not been extracted here:

- [10] High-pressure trickle-bed review (open manuscript linked below): flow regime, wetting, pressure drop and liquid holdup. Highest remaining reactor-engineering priority.
- [11] High-pressure pressure drop/holdup model, and [12] experimental liquid–solid transfer. Obtain full papers if the goal is a rigorous multiphase closure, including pressure-dependent wetting.
- [14] Nickel phosphide clathrates, BaNi₂P₄ and SrNi₂P₄, nitrate hydrogenation: related catalyst only. Optional for catalyst discovery, not a substitute for the binary Ni₂P rate law.
- [15] Huo et al. supporting information for Fig. S1–S4 (catalyst-loading dependence, nitrate-order fit, pH and recycling). Useful if Ru/C becomes a serious alternative.

Search also returned several 2025–2026 nickel-phosphide nitrate *electroreduction* papers; these were excluded because externally applied potential and electrochemical H generation differ from molecular-H₂ hydrogenation. No directly applicable supported Ni₂P brine kinetic dataset was located in this targeted search; absence from this search does not establish nonexistence.

### 11.3 Experiments that literature cannot replace

Obtain measured catalyst loading (ICP and phase characterization), pellet dimensions, apparent and skeletal density, pore volume/distribution, crush strength, dispersion, and Ni/P leaching. Use actual or charge-balanced synthetic Keldgate brine with measured nitrate, chloride, sulphate, phosphate, carbonate alkalinity, TOC, pH, density and viscosity. Design a staged experiment comparing unsupported and supported catalyst in dilute and representative brine. Measure nitrate, nitrite, total ammonia and gas-phase N₂/NO/N₂O at multiple times; measure dissolved H₂ and gas outlet H₂. Vary nitrate concentration, controlled pH, H₂ partial pressure and temperature independently, with mixing/flow sensitivity to detect external transport limits. Repeat and quantify uncertainty; then measure liquid holdup, wetting and pressure drop in the selected bed. Conduct sustained activity and leaching tests. High-pressure hydrogen systems require qualified equipment and engineering safety review.

**Decision:** retain Ni₂P as provisional A4 candidate, not a confirmed industrial catalyst. The literature search has delivered a reproducible dilute-catalyst calibration and usable transport estimates. It has not resolved the core uncertainty of supported Ni₂P activity/selectivity in high-salinity high-nitrate feed. Do not substitute Ru/C kinetic coefficients or report the current model as experimentally validated.

## 12. References — original catalyst studies

[1] Wei, L.; Liu, D.-J.; Rosales, B. A.; Evans, J. W.; Vela, J. Mild and Selective Hydrogenation of Nitrate to Ammonia in the Absence of Noble Metals. *ACS Catalysis* 2020, 10, 3618–3628. https://doi.org/10.1021/acscatal.9b05338 (`wei2020.pdf`).

[2] Wei et al. Supporting Information to [1], 12 pages, especially Table S1 pp. S5–S7, Figs. S4–S8. https://doi.org/10.1021/acscatal.9b05338.s001 (`cs9b05338_si_001.pdf`).

[3] Xin, H.; Liu, Y.; Hu, C.; Lercher, J. A. Electronic impact of Ni₂P nanoparticle size on hydrogenation rates. *Journal of Catalysis* 2021, 401, 129–136. https://doi.org/10.1016/j.jcat.2021.07.017 (`1-s2.0-S0021951721002827-main.pdf`).

[4] Yang, S.; Liang, C.; Prins, R. A novel approach to synthesizing highly active Ni₂P/SiO₂ hydrotreating catalysts. *Journal of Catalysis* 2006, 237, 118–130. https://doi.org/10.1016/j.jcat.2005.10.021 (`1-s2.0-S0021951705004057-main.pdf`).

[5] Lan, X.; Hensen, E. J. M.; Weber, T. Silica-supported Ni₂P: Effect of preparation conditions on structure and catalytic performance in thiophene hydrodesulfurization (HDS). *Catalysis Today* 2017, 292, 121–132. https://doi.org/10.1016/j.cattod.2016.12.040 (`1-s2.0-S0920586116308677-main.pdf`).

Project reference: *UK Nitrate-Rich Wastewater Feedstock to Green Urea Plant Design*, sections on design basis and Cambridge Water brine composition. Prior A4 geometry and bed assumptions retained solely for comparison.

### Additional supplied papers (full text examined)

[6] Torín-Ollarves, G. A.; Trusler, J. P. M. Solubility of hydrogen in sodium chloride brine at high pressures. *Fluid Phase Equilibria* **2021**, *539*, 113025. https://doi.org/10.1016/j.fluid.2021.113025. Tables 3–5, Eqs. 18–22, pp. 6–9. (`1-s2.0-S037838122100087X-main.pdf`)

[7] Piszko, M.; Schmidt, P. S.; Rausch, M. H.; Fröba, A. P. Diffusivities in water or aqueous solutions of sodium chloride with dissolved hydrogen and methane by dynamic light scattering. *International Journal of Hydrogen Energy* **2024**, *68*, 499–510. https://doi.org/10.1016/j.ijhydene.2024.04.217. Eq. 14, Table 1, p. 508. (`1-s2.0-S0360319924015118-main.pdf`)

[8] Iliuta, I.; Larachi, F.; Grandjean, B. P. A.; Wild, G. Gas–liquid interfacial mass transfer in trickle-bed reactors: state-of-the-art correlations. *Chemical Engineering Science* **1999**, *54*, 5633–5645. https://doi.org/10.1016/S0009-2509(99)00129-3. Tables 1, 3–5, nomenclature pp. 5641–5642. (`1-s2.0-S0009250999001293-main.pdf`)

[9] Huo, X.; Vanneste, J.; Cath, T. Y.; Strathmann, T. J. A hybrid catalytic hydrogenation/membrane distillation process for nitrogen resource recovery from nitrate-contaminated waste ion exchange brine. *Water Research* **2020**, *175*, 115688. https://doi.org/10.1016/j.watres.2020.115688. Tables 1–4, Figs. 3, 5, 6, Sections 2–3. (`1-s2.0-S0043135420302244-main.pdf`)

### Further references / retrieval leads (not full-text calibrated here)

[10] Al-Dahhan, M. H.; Larachi, F.; Duduković, M. P.; Laurent, A. High-Pressure Trickle-Bed Reactors: A Review. *Industrial & Engineering Chemistry Research* **1997**, *36*, 3292–3314. https://doi.org/10.1021/ie9700829. Open manuscript: https://crelonweb.eec.wustl.edu/ReactionEngineeringCourse/Day-8.pdf.

[11] Al-Dahhan, M. H.; Duduković, M. P. Pressure drop and liquid holdup in high pressure trickle-bed reactors. *Chemical Engineering Science* **1994**. https://www.sciencedirect.com/science/article/pii/0009250994003157. Also: Prediction of Pressure Drop and Liquid Holdup in High-Pressure Trickle-Bed Reactors, https://doi.org/10.1021/ie970460%2B.

[12] Highfill, W.; Al-Dahhan, M. H. Liquid-Solid Mass Transfer Coefficient in High Pressure Trickle Bed Reactors. *Chemical Engineering Research and Design* **2001**, *79*(6). https://scholarsmine.mst.edu/che_bioeng_facwork/1365/; publisher https://www.sciencedirect.com/science/article/pii/S0263876201720928.

[13] Wakao, N.; Funazkri, T. Effect of fluid dispersion coefficients on particle-to-fluid mass transfer coefficients in packed beds: correlation of Sherwood numbers. *Chemical Engineering Science* **1978**, *33*, 1375–1384. https://doi.org/10.1016/0009-2509(78)85120-3. General engineering reference; not independently re-extracted for this report.

[14] Nitrate and nitroarene hydrogenations catalyzed by alkaline-earth nickel phosphide clathrates. *Dalton Transactions* **2024**. https://doi.org/10.1039/D4DT00332B. Related phase, not binary Ni₂P.

[15] Supporting information associated with [9], available from publisher article landing page https://doi.org/10.1016/j.watres.2020.115688; not included in ZIP.

**Scope note:** The original five catalyst-paper extraction is consolidated from the preceding review (same five PDFs are in the current ZIP); the four new papers were read directly for this update. Numerical correlations are transcribed and illustrative calculations are independently evaluated, but the complete reactor model and neural network predictions have not been validated against original datasets. The current file is a technical evidence dossier, not a final engineering specification.
