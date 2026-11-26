"""Cooling-water and refrigeration allocation for the ammonia loop."""

CW_SUPPLY_C = 30.0
CW_APPROACH_K = 5.0
CW_ELECTRIC_FRACTION = 0.02  # kWe/kW of heat removed; pump/fan estimate


def c1_cooling_fractions(inlet_c, outlet_c):
    """Return (water, chiller) fractions of C1 duty.

    Heat capacity is taken constant over the cooling interval. This split is
    exact at the 40 C design point, where the entire duty uses cooling water.
    """
    limit = CW_SUPPLY_C + CW_APPROACH_K
    if inlet_c <= outlet_c:
        raise ValueError("C1 inlet must be hotter than its outlet")
    if outlet_c < 10.0:
        raise ValueError("C1 outlet below 10 C is outside the R717 design range")
    chilled_drop = max(0.0, min(inlet_c, limit) - outlet_c)
    chilled = chilled_drop / (inlet_c - outlet_c)
    return 1.0 - chilled, chilled
