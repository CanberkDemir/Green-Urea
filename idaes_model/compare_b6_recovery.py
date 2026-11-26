"""Controlled B6 bypass comparison using the existing recovery shortcuts.

NH3 production is held constant; this does not solve the reactor or predict
conversion after changing its feed. No reactor heat credit is assumed.
"""
import csv
import sys

from ammonia_energy import recovery_streams, pinch


def main():
    writer = csv.writer(sys.stdout)
    writer.writerow(["NH3_kgph", "water_kgph", "option", "B6_kWe",
                     "stripper_reboiler_kWth", "rectifier_reboiler_kWth",
                     "recovery_hot_utility_no_HI_kWth",
                     "recovery_hot_utility_HI_kWth", "recovery_shaft_kWe"])
    # Legacy production, hypothetical high-conversion Ni2P production,
    # and the current low-conversion Ni2P scenario, respectively.
    for nh3 in (11.96, 20.73, 0.222):
        for water in (1400.0, 5000.0, 7621.015):
            b6 = (7621.015 - water) / 1000 * (35 if water == 1400 else 24)
            for option in ("S1", "S2", "S3"):
                streams, shaft, _ = recovery_streams(
                    option, nh3, water, water + 879.0 + nh3, 40.0)
                result = pinch(streams, 10.0)
                duties = {s.name: s.duty_kw for s in streams}
                writer.writerow([nh3, water, option, round(b6, 3),
                                 round(duties.get("stripper reboiler", 0), 3),
                                 round(duties["rectifier reboiler (15 bar)"], 3),
                                 round(result.q_hot_no_hi, 3),
                                 round(result.q_hot_min, 3), round(shaft, 3)])


if __name__ == "__main__":
    main()
