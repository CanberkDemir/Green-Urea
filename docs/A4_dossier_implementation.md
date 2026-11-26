# A4 dossier implementation

Implemented the Table S1 fixed-intercept log fit, retaining the nitrate and ammonium measurements and unassigned nitrogen explicitly in `ni2p_dossier.py`. The unassigned nitrogen is not interpreted as irreversible N2. The apparent fit is not intrinsic kinetics.

The PFR now uses the fitted rate, 12.39% equivalent active loading, and the extended Krichevsky–Kasarnovsky NaCl solubility correlation. Gas composition uses a pure-water saturated-vapour approximation and ideal H2 fugacity; water mass per solution volume converts molality to concentration. The laboratory reference uses diluted wet hydrogen at one atmosphere. Mixed-salt activities and pressure-dependent fugacity remain approximations.

The module also provides the dossier diffusivity equation and transcribed Iliuta neural correlation. The latter returns normalized inputs for domain checks but is not automatically coupled to the PFR: temperature and diameter are outside its database, and viscosity, surface tension, gas recycle, wetting and pressure drop remain unspecified. Its coefficients have not been independently checked against the source paper. Solubility and diffusivity reproduce the dossier numerical examples.

Default PFR run: 584 kg active equivalent, inlet dissolved-H2 equilibrium 13.982 mol/m3, 1.25% nitrate conversion, 0.259 kg/h NH3 equivalent; hydrogen and nitrogen residuals below 1e-10 mol/s. Four PFR conservation/limiting-case tests pass. Fixed 96% selectivity remains an explicitly labelled endpoint scenario, not a fit to transient ammonium observations. Apparent-rate extrapolation and fixed transfer assumptions still control the result.

Still required: jointly fit a defensible transient nitrogen network; implement pH/speciation and acid/base balances, validated hydrodynamics and pellet diffusion, and integrate into the IDAES flowsheet. Acid equivalents already reported by the PFR are stoichiometric demand, not controlled pH or dosing. A4 remains WIP. No Ru/C or phenol kinetic constants have been transferred to Ni2P.
