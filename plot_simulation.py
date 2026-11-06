"""Plot simulated cases and rainfall per facility as small multiples: ``python plot_simulation.py <simulated.csv> <out.png>``."""

import sys

import matplotlib.pyplot as plt
import pandas as pd

from simulator import period_start, periods_per_year

CASES, RAIN, INK, MUTED, GRID = "#2a78d6", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"

df = pd.read_csv(sys.argv[1])
df["date"] = pd.to_datetime([period_start(str(p)) for p in df["time_period"]])
weekly = periods_per_year(str(df["time_period"].iloc[0])) != 12
unit = "week" if weekly else "month"
locations = sorted(df["location"].unique())
fig, axes = plt.subplots(len(locations), 2, figsize=(13, 1.35 * len(locations) + 0.8), sharex=True,
                         gridspec_kw={"width_ratios": [2.2, 1]})
for row, location in zip(axes, locations):
    sub = df[df["location"] == location]
    row[0].plot(sub["date"], sub["disease_cases"], color=CASES, lw=1.0 if weekly else 1.4, marker="o", ms=1.2 if weekly else 2.5)
    row[1].bar(sub["date"], sub["rainfall"], width=6 if weekly else 25, color=RAIN)
    pop = sub["population"].iloc[0]
    row[0].set_title(f"{location}  (catchment {pop:,.0f})", loc="left", fontsize=9, color=INK, pad=3)
    for ax in row:
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_ylim(bottom=0)
axes[0, 0].text(0, 1.55, f"Disease cases per {unit}", transform=axes[0, 0].transAxes, fontsize=11, weight="bold", color=INK)
axes[0, 1].text(0, 1.55, f"Rainfall (mm per {unit})", transform=axes[0, 1].transAxes, fontsize=11, weight="bold", color=INK)
fig.patch.set_facecolor("#fcfcfb")
fig.tight_layout(h_pad=0.9)
fig.savefig(sys.argv[2], dpi=130)
