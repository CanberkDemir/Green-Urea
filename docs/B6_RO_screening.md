# B6 reverse-osmosis screening

Run: `conda run -n idaes-env python idaes_model/ro_screening.py`.
Results: `B6_RO_screening.csv`. This calculation ran on 2026-11-26.
The named environment exists but does not currently import `idaes`.

The feed contains 7621.015 kg/h water and 878.985 kg/h listed ions.
All ions are retained in this screening; nitrate leakage is not predicted.
At 25 C the reference osmotic pressure is approximately 92.8 bar,
using pi = RT rho_water phi sum(ion molality), with phi=1.
This is a solvent-activity approximation, not calibrated high-salinity
electrolyte thermodynamics. Phi=0.9 and 1.2 are sensitivity cases, not
confidence bounds. Concentration polarization multiplies bulk osmotic
pressure by an assumed 1.10. Feed pressure additionally includes 10 bar
net driving pressure, 2 bar channel loss and 1 bar permeate pressure.
No membrane area or throughput guarantee follows without permeability data.

At 83 bar none of these cases achieves recovery with the specified margin.
At 120 bar, phi=1 removes 350 kg/h water (4.6% of feed water); phi=0.9
removes 1077 kg/h (14.1%); phi=1.2 gives no recovery at the stated margin.
Leaving 5000 kg/h water requires 153–200 bar across these sensitivities,
169 bar at phi=1. The original 1400 kg/h target is precipitation-invalid;
its formal pressure extrapolation in the CSV is not a feasible design.

Power assumes 1 bar feed, 80% pump efficiency, feed density 1070 kg/m3,
concentrate density 1100 kg/m3, and an optional 80% efficient hydraulic
energy recovery device on concentrate depressurization to 27 bar for
the reactor. At 120 bar: 32.8 kWe without energy recovery, 17.8 kWe with
recovery in the reference case. This includes feed pressurization to the
reactor pressure; a comparison with existing pressurized feed must subtract
the corresponding baseline pumping load. Pretreatment and auxiliaries are
excluded. The densities and efficiencies are explicit estimates.

At fixed 11.96 kg/h NH3, the existing vacuum recovery shortcut estimates
309 kWth heating with no B6, versus 295 kWth after the reference 120 bar RO
case (internal separation heat recovery included). The optimistic phi=0.9
case gives 267 kWth. Reactor conversion, heat release, salt effects on NH3
speciation, membrane nitrate leakage and cooling electricity are not solved.
Thermal duty savings cannot be directly subtracted from electrical input.

Conclusion: ordinary seawater RO is pressure constrained here. A 120 bar
unit offers uncertain, modest concentration; it does not establish a low
energy replacement for the original B6. A higher pressure design requires
validated water activities, membrane permeability/rejection and pressure
ratings, plus full scaling and downstream reactor/recovery calculations.

Manufacturer reference: [DuPont FilmTec technical guidance, June 2026](https://www.dupont.com/content/dam/water/amer/us/en/water/public/documents/en/RO-FilmTec-Additional-Information-TechFact-45-D04931-en.pdf).
It lists 83 bar seawater and 120 bar specialised elements and recommends
composition-specific projection rather than a universal TDS limit.
