"""Break-even price bar chart (PDF). Superseded by make_all_paper_figures.py;
kept as a convenience entry point that computes from the solution CSVs."""
from make_all_paper_figures import (
    CASES,
    OUT_DIR,
    UREA_MARKET_GBP_PER_KG,
    _simple_bar,
    case_summary,
)

summaries = [case_summary(name, key, path) for name, key, path in CASES]
_simple_bar(
    [s["bep_gbp_per_kg"] for s in summaries],
    [s["name"] for s in summaries],
    r"Break-even price (GBP kg$_{\mathrm{urea}}^{-1}$)",
    OUT_DIR / "bep.pdf",
    market=UREA_MARKET_GBP_PER_KG,
    market_label="market price 1.83 GBP/kg",
)
print(f"Wrote {OUT_DIR / 'bep.pdf'}")
