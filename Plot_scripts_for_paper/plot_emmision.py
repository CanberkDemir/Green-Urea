"""Gate-to-gate emissions bar chart (PDF). Superseded by
make_all_paper_figures.py; kept as a convenience entry point."""
from make_all_paper_figures import CASES, CONVENTIONAL_REFS, OUT_DIR, _simple_bar, case_summary

summaries = [case_summary(name, key, path) for name, key, path in CASES]
names = [s["name"] for s in summaries] + [n for n, _ in CONVENTIONAL_REFS]
values = [s["emissions_kgco2_per_kg"] for s in summaries] + [v for _, v in CONVENTIONAL_REFS]
_simple_bar(
    values,
    names,
    r"CO$_2$ emissions (kg$_{\mathrm{CO_2}}$ kg$_{\mathrm{product}}^{-1}$)",
    OUT_DIR / "emissions.pdf",
)
print(f"Wrote {OUT_DIR / 'emissions.pdf'}")
