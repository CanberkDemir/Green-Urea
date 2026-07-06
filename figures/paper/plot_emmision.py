"""Gate-to-gate emissions comparison chart (PDF). Superseded by
make_all_paper_figures.py; kept as a convenience entry point."""
from make_all_paper_figures import CASES, OUT_DIR, case_summary, fig_emissions

summaries = [case_summary(name, key, path) for name, key, path in CASES]
fig_emissions(summaries, OUT_DIR / "emissions.pdf")
print(f"Wrote {OUT_DIR / 'emissions.pdf'}")
