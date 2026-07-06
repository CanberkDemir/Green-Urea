"""Annualized-cost bar chart (PDF). Superseded by make_all_paper_figures.py;
kept as a convenience entry point that computes from the solution CSVs."""
from make_all_paper_figures import CASES, OUT_DIR, _simple_bar, case_summary

summaries = [case_summary(name, key, path) for name, key, path in CASES]
_simple_bar(
    [s["annualized_cost_gbp_per_y"] / 1e6 for s in summaries],
    [s["name"] for s in summaries],
    "Annualized cost (million GBP y$^{-1}$)",
    OUT_DIR / "annualized_cost_cases.pdf",
)
print(f"Wrote {OUT_DIR / 'annualized_cost_cases.pdf'}")
