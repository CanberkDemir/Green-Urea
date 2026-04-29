import matplotlib.pyplot as plt
import numpy as np

cases = ["Grid only", "10% Grid", "5% Grid", "Wind only"]

# Wind capacity (W_cap)
W_cap = [
    13.010433,     # Grid only
    749.344394,    # 10% Grid
    789.129526,    # 5% Grid
    851.036511     # Wind only
]

# Electrolyzer capacity (E_cap)
E_cap = [
    377.869930,
    445.430554,
    460.306995,
    486.881492
]

# Battery / buffer capacity (B_cap)
B_cap = [
    551.231561,
    6808.930209,
    10000.000000,
    10000.000000
]

# Hydrogen storage capacity (H_cap)
H_cap = [
    0.005833,
    162.048637,
    101.267539,
    983.435643
]

# Ammonia storage capacity (NH3_cap)
NH3_cap = [
    0.0,
    239.253690,
    622.299565,
    801.933401
]

x = np.arange(len(cases))
width = 0.25

# Use the viridis colormap for the plotted capacities.
viridis = plt.get_cmap("viridis")
color_H, color_NH3, color_B = [viridis(point) for point in (0.15, 0.5, 0.85)]

fig, ax1 = plt.subplots(figsize=(9,5))

# ---- LEFT AXIS ----
bars_H = ax1.bar(x - width, H_cap, width, color=color_H, label="Hydrogen Storage Capacity")
bars_NH3 = ax1.bar(x, NH3_cap, width, color=color_NH3, label="NH3 Storage Capacity")

ax1.set_ylabel("Hydrogen Storage & Ammonia Capacity")
ax1.set_xticks(x)
ax1.set_xticklabels(cases)

# ---- RIGHT AXIS ----
ax2 = ax1.twinx()
bars_B = ax2.bar(x + width, B_cap, width, color=color_B, label="Battery Capacity")

ax2.set_ylabel("Battery / Buffer Capacity", color=color_B)

# Color the right axis ticks and spine
ax2.tick_params(axis='y', colors=color_B)
ax2.spines['right'].set_color(color_B)

# ---- LEGEND ----
handles = [bars_H, bars_NH3, bars_B]
labels = ["H_cap", "NH3_cap", "B_cap"]
ax1.legend(handles, labels, loc="upper left", frameon=False)

# ---- TITLE ----
plt.tight_layout()
plt.show()
