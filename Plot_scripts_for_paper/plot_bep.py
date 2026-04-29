import matplotlib.pyplot as plt

VIRIDIS_015 = "#463480"

cases = ["Grid only", "10% Grid", "5% Grid", "Wind only"]
annualized_costs = [431.93, 1292.79, 1338.83, 1412.54]

plt.figure(figsize=(8, 5))

bars = plt.bar(cases, annualized_costs, color=VIRIDIS_015)

plt.ylabel("Break Even Prices (£/kg$_{urea}$)")

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.1f}",
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()
