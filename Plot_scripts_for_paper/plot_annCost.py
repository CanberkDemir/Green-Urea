import matplotlib.pyplot as plt

VIRIDIS_015 = "#463480"

# Data from your PDF
cases = [
    "Grid only",
    "10% Grid",
    "5% Grid",
    "Wind only"
]

annualized_costs = [
    1.581110817950e7,
    4.698385350954e7,
    4.867219617765e7,
    5.133907541000e7
]

# Create figure
plt.figure(figsize=(8, 5))

# Bar plot
bars = plt.bar(cases, annualized_costs, color=VIRIDIS_015)

# Labels and title
plt.ylabel("Annualized Cost (£/year)")

# Optional: format y-axis in millions
plt.ticklabel_format(style='sci', axis='y', scilimits=(6,6))

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f'{height/1e6:.2f}M',
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()
