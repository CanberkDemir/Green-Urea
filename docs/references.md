# Literature references for IDAES modelling decisions

Every modelling decision that depends on outside data cites its source here. Each entry
gives the reference, what it says (quantitatively where possible), which TODO item or
code it supports, and how confident we are in our reading of it. Entries marked *(not
read in full)* were assessed from abstracts or search summaries only. Check the full
text before quoting numbers from them in the manuscript.

---

## 1. Nitrate hydrogenation catalysts and selectivity (TODO A2)

**[R1] Huo, X.; Van Hoomissen, D. J.; Liu, J.; Vyas, S.; Strathmann, T. J. (2017).**
"Hydrogenation of aqueous nitrate and nitrite with ruthenium catalysts."
*Applied Catalysis B: Environmental*.
<https://www.sciencedirect.com/science/article/abs/pii/S0926337317303508> *(not read in full)*
- Ru (carbon- and alumina-supported) reduces **nitrate selectively to ammonium, with no aqueous nitrite intermediate observed**. It is active at ambient temperature and H2 pressure.
- Nitrite hydrogenation, by contrast, gives a mixture of NH4+ and N2, shifting towards N2 at higher nitrite concentration.
- **Supports:** a high-NH3-selectivity catalyst exists for this chemistry, but it is Ru, not Pd–Cu.

**[R2] Huo, X.; Vanneste, J.; Cath, T. Y.; Strathmann, T. J. (2020).**
"A hybrid catalytic hydrogenation/membrane distillation process for nitrogen resource recovery from nitrate-contaminated waste ion exchange brine."
*Water Research* 175.
<https://www.sciencedirect.com/science/article/abs/pii/S0043135420302244> *(not read in full)*
- Ru/C hydrogenates nitrate with **high selectivity to ammonia under conditions representative of waste brines**. The ammonia is then recovered by membrane distillation as a fertilizer product.
- **Supports:** this is the closest published analogue to our nitrate-brine → NH3 → recovery concept. It is a strong citation for the rewrite, and its MD recovery step is an alternative to the gas-stripping membrane in TODO B2.

**[R3] Sustainable catalytic hydrogenation of high concentration nitrate to ammonia over ruthenium-based catalysts (2024/2025).**
*Separation and Purification Technology*.
<https://www.sciencedirect.com/science/article/abs/pii/S1383586624030144> *(not read in full; author list not yet recorded)*
- Ru–Pd/CeO2 reports **near-100% ammonia selectivity** across nitrate concentrations.
- **Supports:** the A2 option of a high-selectivity network.

**[R4] Pd–Cu literature for drinking-water denitrification (Pd–Cu is N2-selective).**
- Pd–Cu catalysts were developed to convert nitrate to **N2**; NH4+ is an unwanted by-product. Nitrate reduction occurs at Pd–Cu sites and nitrite reduction at Pd sites. Pd-rich surfaces give low nitrite and **high N2 selectivity**.
  Catalytic reduction of nitrates and nitrites on pumice-supported Pd–Cu:
  <https://www.researchgate.net/publication/223444274_Catalytic_reduction_of_nitrates_and_nitrites_in_water_solution_on_pumice-supported_Pd-Cu_catalysts> *(not read in full)*
- At **alkaline pH (10.5–12.5)** Pd–Cu stops largely at **nitrite (up to 93% nitrite selectivity)**, with the rest as ammonium.
  Catalytic stepwise nitrate hydrogenation in batch-recycle fixed-bed reactors, *J. Hazard. Mater.* 2007:
  <https://pubmed.ncbi.nlm.nih.gov/17478039/> *(not read in full)*
  Selective hydrogenation of nitrate to nitrite over Cu–Pd clusters on activated carbon:
  <https://www.sciencedirect.com/science/article/abs/pii/S1381116906000719> *(not read in full)*
- **Implication:** the manuscript's ~90% NH3 selectivity on Pd–Cu (citing Zhang et al., `\cite{Pd-Cu_experimental}`) is **not supported** by the Pd–Cu literature found here. The `.bkp` network (55%) is closer to Pd–Cu behaviour. The rewrite should either switch the catalyst to Ru (R1–R3) and keep a high-selectivity network, or keep Pd–Cu with low selectivity. **Team decision; not made in the code.** Both are available via `reactions/ammonia_reactions.py: NETWORK = "bkp" | "si"`.
- **To do:** pull the full text of the paper's Zhang et al. reference and check what selectivity it reports and under which conditions.

## 2. Ammonia recovery from the hydrous product (TODO B1–B3)

**[R5] Iddya, A.; Hou, D.; Khor, C. M.; Ren, Z.; Tester, J.; Posmanik, R.; Gross, A.; Jassby, D. (2020).**
"Efficient ammonia recovery from wastewater using electrically conducting gas stripping membranes."
*Environmental Science: Nano* 7, 1759–1771. DOI 10.1039/C9EN01303B.
<https://pubs.rsc.org/en/content/articlelanding/2020/en/c9en01303b>
- **7.1 ± 1.1 kWh/kg N** of electricity. Water electrolysis at the membrane generates OH−, which converts NH4+ to NH3, which is then stripped through the membrane. Flux 68.8 ± 8.0 g N/m²/day.
- Context quoted alongside it: Haber–Bosch 8.9–19.3 kWh/kg N; nitrification–denitrification 2.3–6.5 kWh/kg N; membrane distillation ≈2.2 kWh/kg N.
- **Supports:** the paper's `e_membrane`. Note that this is **electricity with no temperature level**. It cannot be "heat-recovered" as the manuscript's `Q_rec ≤ Q_S` assumes (see TODO E1).

**[R6] Henry's-law solubility of NH3 in water: Sander, R. (2015).**
"Compilation of Henry's law constants (version 4.0) for water as solvent."
*Atmospheric Chemistry and Physics* 15, 4399–4981.
- NH3: H ≈ 59 mol/(kg·bar) at 25 °C, d ln H / d(1/T) ≈ 4200 K.
- **Supports:** `ammonia_energy.py: henry_k()` for the shortcut stripper (B3). Value quoted from memory of the compilation. **Verify the exact entry before publication.**

## 3. Brine concentration energy (TODO A9)

**[R7] Practical limits and energy of pressure-driven brine concentration.**
- Conventional RO is limited to about **70–80 g/L brine** (membrane burst pressure ~70–80 bar).
  Gradiant CFRO white paper (2023): <https://gradiant.com/wp-content/uploads/2023/04/Gradiant-CFRO-whitepaper.pdf>
- Counter-flow RO reaches **110–130 g/L** at **≈5.8 kWh/m3 permeate** (50% recovery). Same source.
- Osmotically assisted RO at a 70 g/L feed: **6.37 kWh/m3** at 44% recovery; pressure-based processes can reach about 125 g/L.
  Peters & Hankins (Oxford): <https://ora.ox.ac.uk/objects/uuid:84559056-508b-4f68-aaff-63197ab64538/files/m7eb369be2d057f96a1a3172c1ec5ccdc> *(not read in full)*
- Mechanical vapour compression for 70 → 250 g/L: **≈24 kWh/m3**.
  Univ. Padova thesis (Tartarelli): <https://thesis.unipd.it/bitstream/20.500.12608/79841/1/Tartarelli_Serena.pdf> *(not read in full)*
- **Implication:** our NO3-IN brine is already about 103 g/kg dissolved solids, above the conventional RO limit. B6 concentrates it to about **38 wt% salts**, which is **above NaCl saturation (~26.4 wt% at 25 °C)**. That is not a feasible single-phase solution: salts would crystallize. A realistic B6 is thermal (MVC/evaporator-crystallizer) at roughly 25–40 kWh/m3 of water removed. **That would add about 180–280 kW for ~7 m3/h**, the same order as the whole ammonia section. See A9.

## 4. Urea / DEF

**[R8] Pustjens, J. H. M.; Van Den Tillaart, J. A. A. — US 10,759,745 B2 (Stamicarbon).**
"Urea production process and plant" for a purified urea solution, e.g. DEF.
<https://patents.google.com/patent/US10759745B2/en>
- No HP stripper; reactor at 150–250 bar, >190 °C, N/C 2.7–5.0; MP dissociation at 15–25 bar and LP at 3–5 bar; low-pressure steam stripping at 0.01–1.5 bar(abs) with condensate recycled to the reactor; DEF at 32.5 wt% (ISO 22241: alkalinity as NH3 < 0.2 wt%, biuret ≤ 0.3 wt%).
- **Supports:** `def_lowp_flowsheet.py` (TODO D1–D4). The patent gives no conversion, residence time or energy numbers.

## 5. Thermodynamic and solver notes (not literature, but recorded for the same reason)
- idaes-ext 3.4.2 (Windows x86-64): any two-phase **SRK** state block crashes IPOPT with access violation 0xC0000005, including IDAES's own BT_PR example switched to SRK. PR works. See TODO C5.
