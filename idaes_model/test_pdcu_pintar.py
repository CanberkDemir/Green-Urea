"""Check the published nitrate rate against Pintar et al. Table 3."""

from reactions.pdcu_pintar import TABLE_2, nitrate_disappearance_rate


def test_table_3_initial_rate():
    # Table 3: 293 K, 100 mg/L NO3, 0.46 bar H2, 0.15 g/L catalyst.
    # Reported rate is 15.2 mg/(g_cat min) measured, 15.0 calculated.
    rate_per_catalyst = nitrate_disappearance_rate(
        100.0, 0.46, 0.15, TABLE_2[293.0]
    ) / 0.15
    assert abs(rate_per_catalyst - 15.0) < 0.2


def test_zero_hydrogen_and_nitrate():
    assert nitrate_disappearance_rate(100.0, 0.0, 0.15) == 0.0
    assert nitrate_disappearance_rate(0.0, 0.46, 0.15) == 0.0


if __name__ == "__main__":
    test_table_3_initial_rate()
    test_zero_hydrogen_and_nitrate()
    print("Pintar nitrate-rate checks passed")
