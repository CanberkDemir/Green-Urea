# Direct electrochemical nitrate reduction: survey and screening

Model: `idaes_model/electro_no3rr.py` (algebraic cascade flow-cell, no Pyomo).

## Why consider it
The current route turns electricity into H2 (1.23 V reversible, ~1.9 V at
50 kWh/kg H2), compresses it to 27 bar, and hydrogenates nitrate over
Ni2P / Pd-Cu. Direct NO3RR uses the same 8 e- per NH3 but at a reversible
cell voltage of only ~0.54 V (NO3-/NH3 +0.69 V vs RHE, OER 1.23 V). It also
removes the H2 compressor, the 27 bar reactor, the flash/recycle and the
R717 chiller, and the catholyte leaves at pH ~13, so NH3 is already free
for stripping.

## Literature state (Nov 2026)
| Topic | What is reported | Source |
|---|---|---|
| NH3 from nitrate, high rate | FE ~100% at ~1-2 A/cm2 in 1 M nitrate lab flow cells (Ru/Cu2O, Fe2O3/Fe-N-C) | [OSTI, JACS Ru/Cu](https://www.osti.gov/pages/biblio/2432566-ammonia-electrosynthesis-from-nitrate-using-rutheniumcopper-cocatalyst-system-full-concentration-range-study) |
| Membrane-free alkaline cell | 4.22 A/cm2 NH3 partial current, FE 84.5%, direct NH3 solution / NH4HCO3 | [Chen et al., NSF PAR](https://par.nsf.gov/servlets/purl/10408531) |
| Dilute feed (2000 ppm NO3-) | FE > 90% at > 100 mA/cm2 with a solid-electrolyte, three-chamber cell | [AIChE 2024](https://proceedings.aiche.org/conferences/aiche-annual-meeting/2024/proceeding/paper/707g-direct-electrosynthesis-ammonia-nitrate-reduction-reaction-catalyst-design-and-electrolyzer) |
| Energy efficiency | best half-cell EE ~40%; FE 98% at -0.05 V vs RHE | [Li et al. 2025, PolyU](https://research.polyu.edu.hk/en/publications/a-reaction-diffusion-coupled-strategy-for-ampere-level-electrocat/) |
| Scale-up / TEA | scaled membrane flow electrolyzers; TEAs below $1.15/kg NH3 in some cases; energy efficiency and catalyst life dominate; dilute feeds and enrichment are the main barrier | [RSC EES 2025](https://pubs.rsc.org/en/content/articlepdf/2025/ee/d4ee04382k), [Green Chem 2025 review](https://pubs.rsc.org/en/content/articlelanding/2025/gc/d5gc04166j), [EY 2023](https://pubs.rsc.org/en/content/articlepdf/2023/ey/d3ey00058c) |
| Urea from CO2 + NO3- | FE typically tens of %, low current density, many parallel pathways; acknowledged gap to industrial needs | [Muhyuddin et al. EES 2024](https://pubs.rsc.org/en/content/articlehtml/2024/ee/d4ee00561a), [Trans. Tianjin Univ. 2025](https://link.springer.com/10.1007/s12209-025-00444-2), [Nano Research 2026](https://www.sciopen.com/local/article_pdf/10.26599/NR.2026.94908662.pdf) |

Gap: no paper found that runs NO3RR in a ~1.6 M chloride brine. The design
below handles chloride by keeping it on the cathode side of a CEM, which is
an assumption, not demonstrated.

## Our feed
8.02 m3/h brine, 0.158 M NO3- (≈ 9.8 g/L), ~1.6 M NaCl, max 21.6 kg/h NH3.
That nitrate level is close to the 0.1 M used in most lab work, so the
feed is better suited to NO3RR than typical wastewater (mM). Limiting current
at the inlet is only ~0.1-0.6 A/cm2 (k_m 1-5e-5 m/s), and it falls with
conversion. That is why the model uses a staged cascade.

Concentrating in B6 first does not help: it pushes NaCl past saturation
(6.7 M vs 6.1 M).

## Results (95% conversion, 4 stages, raw brine)
| Case | Area m2 | FE | Power kW | kWh/kg NH3 |
|---|---|---|---|---|
| optimistic | 141 | 0.83 | 518 | 25.3 |
| central | 389 | 0.86 | 573 | 27.9 |
| pessimistic | 979 | 0.80 | 744 | 36.2 |
| central, 1 stage | 348 | 0.43 | 1172 | 57.1 |

Staging roughly halves the energy, because a single cell runs mostly
mass-transfer limited and wastes current on HER.

The NH3 still needs recovery from the pH-13 catholyte. Use the existing
`ammonia_energy.py` options (vacuum stripper / membrane). No lime or NaOH is
needed, since the cathode already makes the OH-.

## Direct urea (CO2 + NO3-)
At the same nitrate (36 kg/h urea): optimistic 33, central 79, pessimistic
271 kWh/kg urea, with 430 to 26,000 m2 of electrode. The bigger problem is
the product: ~0.3 wt% urea in 1.6 M NaCl. Urea is non-volatile and cannot
be stripped, so recovering it costs about as much as evaporating the brine.
This route is TRL 1-2 and is not realistic at this scale today.

## Comparison with the current route (ammonia_energy.py, Ft=8500, Fh2=10)
| | Current: H2 + hydrogenation | Direct NO3RR (central) |
|---|---|---|
| NH3 made | 11.96 kg/h (55% of nitrate) | 20.5 kg/h (95%) |
| Electrolysis | 500 kW, 41.8 kWh/kg NH3 | none |
| Reaction section | incl. B6 217.7 kW and H2 compression | 573 kW cells, 27.9 kWh/kg; no B6 |
| NH3 recovery | vacuum stripper + HI: 24.0 kWh/kg total section | stripper on 8 m3/h pH-13 brine; ~3.7x the vapour of S2, but the reboiler can run on cell waste heat (~300 kW at 50-60 C) |
| Total | ~66 kWh/kg NH3 | 39.5 / 39.6 / 45.0 kWh/kg NH3 (opt/central/pess; plant_energy.run_electro_case, cells at 65 C, S2 shortcut stripper, pinch + heat pump) |

Caveats: chloride-brine NO3RR, catalyst life in 1.6 M Cl-, and the CEM
H+/Na+ split are not demonstrated in the literature. The cathode
overpotential (0.7-1.2 V) is the largest single loss.

Integrated result (`plant_energy.py --electro-only`, ammonia section): 345 kg/h
stripping vapour; 285-490 kW of cell heat is recovered, but heating 8.5 t/h
brine to 65 C leaves 180-350 kW hot utility (pinch 42-60 C). Running the cells
colder, or a feed/effluent exchanger with a smaller dT_min, is the next lever.
The DEF (urea) section currently fails to solve, so per-kg-urea totals are pending.
