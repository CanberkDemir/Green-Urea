"""B6 RO pressure/energy screening; not a calibrated membrane simulation.

Ideal osmotic-coefficient reference with sensitivity; full solute retention.
Pressure margin includes permeate pressure, channel loss and driving pressure.
"""
import csv
import sys
from ammonia_energy import recovery_streams, pinch

WATER = 7621.015
SALTS = {"Na": (330.565, 22.98977), "Cl": (453.9, 35.453),
         "NO3": (78.625, 62.0049), "SO4": (15.895, 96.0626)}
ION_MOL = sum(m / mw * 1000 for m, mw in SALTS.values())
SALT_MASS = sum(m for m, _ in SALTS.values())


def evaluate(water, phi=1.0):
    # pi = -RT ln(a_w)/Vw, ln(a_w) = -phi*Mw*sum(m_i).
    pi = 0.083144626 * 298.15 * 0.997 * ION_MOL / water
    pressure = phi * pi * 1.10 + 10 + 2 + 1
    # Assumed densities, feed 1 bar; concentrate delivered at 27 bar.
    # No recovery of the retained 27 bar is credited.
    q_feed = (WATER + SALT_MASS) / 1070
    q_brine = (water + SALT_MASS) / 1100
    gross = q_feed * (pressure - 1) / (36 * 0.80)
    recovered = q_brine * max(pressure - 2 - 27, 0) * 0.80 / 36
    streams, _, _ = recovery_streams("S2", 11.96, water,
                                    water + SALT_MASS + 11.96, 40)
    heat = pinch(streams, 10).q_hot_min
    return [water, WATER-water, pi*phi, pressure, gross,
            gross-recovered, heat]


def main():
    writer = csv.writer(sys.stdout)
    writer.writerow(["case", "osmotic_coefficient", "water_left_kgph",
                     "water_removed_kgph", "bulk_osmotic_bar",
                     "feed_pressure_bar", "pump_kWe_no_ERD",
                     "net_kWe_with_ERD", "vacuum_recovery_heat_kWth"])
    for phi in (0.9, 1.0, 1.2):
        for pressure in (83, 120):
            required_water = evaluate(WATER, phi)[2] * WATER * 1.1 / (pressure-13)
            if required_water >= WATER:
                writer.writerow([f"{pressure} bar: no recovery at specified margin", phi])
            else:
                writer.writerow([f"{pressure} bar", phi] +
                                [round(x, 3) for x in evaluate(required_water, phi)])
        for water in (5000, 1400):
            writer.writerow(["target (1400 precipitation-invalid)", phi] +
                            [round(x, 3) for x in evaluate(water, phi)])


if __name__ == "__main__":
    main()
