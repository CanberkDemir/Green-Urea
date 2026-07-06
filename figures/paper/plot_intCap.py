"""Storage-capacity comparison chart (PDF). Superseded by
make_all_paper_figures.py; kept as a convenience entry point."""
from make_all_paper_figures import CASES, OUT_DIR, case_summary, fig_storage_capacities

summaries = [case_summary(name, key, path) for name, key, path in CASES]
fig_storage_capacities(summaries, OUT_DIR / "storage_capacity_cases.pdf")
print(f"Wrote {OUT_DIR / 'storage_capacity_cases.pdf'}")
