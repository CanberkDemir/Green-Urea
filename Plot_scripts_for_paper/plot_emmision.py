import matplotlib.pyplot as plt

VIRIDIS_015 = "#463480"

cases = ["Grid only", "10% Grid", "5% Grid", "Wind only", "Steam Cracking", "Bosch-Meiser"]
emmisions = [15.05, -0.21, -0.74, -1.27, 2.2, 1.83]

plt.figure(figsize=(8, 5))

bars = plt.bar(cases, emmisions, color=VIRIDIS_015)

plt.ylabel(r"$\mathrm{CO_2\ Emissions\ (kg_{CO_2}\,kg_{compound}^{-1})}$")

# Add value labels
for bar in bars:
    height = bar.get_height()
    
    if height >= 0:
        y = height + 0.0     # slightly above bar
        va = 'bottom'
    else:
        y = height - 0.0     # slightly below bar
        va = 'top'
    
    plt.text(
        bar.get_x() + bar.get_width()/2,
        y,
        f"{height:.2f}",
        ha='center',
        va=va
    )

plt.tight_layout()
plt.show()
