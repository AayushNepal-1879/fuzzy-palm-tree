"""
BHU Cross-Match — Static Comparison Graph
==========================================
GitHub Codespaces version — saves to file, no display needed.

Output:
  output/bhu_comparison_graph.png   3-panel comparison graph

Run:
    python bhu_graph.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT = "output"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1. DATA & PHYSICS
# ─────────────────────────────────────────────────────────
z_desi = np.array([0.10, 0.30, 0.51, 0.71, 0.93, 1.32, 2.33])
H_desi = np.array([69.0, 80.5, 90.0, 99.0, 115.5, 150.1, 224.6])
H_err  = np.array([2.0,  2.0,  1.9,  2.1,  3.0,   4.5,   8.0])

def get_H_empirical(z):
    return float(np.interp(z, z_desi, H_desi))

H0, Om, OL, lSE, zQ = 68.0, 0.31, 0.69, 0.085, 1.8

def get_H_theory(z):
    Mz     = np.exp(-lSE * z * np.exp(-z / zQ))
    eff_OL = OL * (1.0 / Mz) ** 2
    return H0 * np.sqrt(Om * (1 + z)**3 + eff_OL)

def get_H_lcdm(z):
    return H0 * np.sqrt(Om * (1 + z)**3 + OL)

# ─────────────────────────────────────────────────────────
# 2. INTEGRATE (normalised DT — same fix as simulation)
# ─────────────────────────────────────────────────────────
z_start = 2.33
a_start = 1.0 / (1.0 + z_start)
DT      = 0.0001     # normalised by H0

print("Integrating physics...")
a_A_hist, a_B_hist = [a_start], [a_start]
z_A, z_B = z_start, z_start
a_A, a_B = a_start, a_start

for _ in range(10_000_000):
    a_A += a_A * (get_H_empirical(z_A) / H0) * DT;  z_A = 1.0 / a_A - 1.0
    a_B += a_B * (get_H_theory(z_B)    / H0) * DT;  z_B = 1.0 / a_B - 1.0
    a_A_hist.append(a_A); a_B_hist.append(a_B)
    if a_A >= 1.0 or a_B >= 1.0:
        break

a_A_hist  = np.array(a_A_hist)
a_B_hist  = np.array(a_B_hist)
steps     = len(a_A_hist)
deviation = np.abs(a_A_hist - a_B_hist) / a_A_hist * 100
idx       = np.linspace(0, steps - 1, 2000, dtype=int)
print(f"Done — {steps:,} steps")

z_range  = np.linspace(0.05, 2.5, 300)
H_emp_c  = np.array([get_H_empirical(z) for z in z_range])
H_theo_c = np.array([get_H_theory(z)    for z in z_range])
H_lcdm_c = np.array([get_H_lcdm(z)     for z in z_range])

# ─────────────────────────────────────────────────────────
# 3. PLOT
# ─────────────────────────────────────────────────────────
COLOR_A = "#10B981"
COLOR_B = "#378ADD"
COLOR_D = "#E24B4A"
COLOR_L = "#888888"
BG      = "#0d0d0f"
GRID    = "#2a2a35"

def style(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors="#666666", labelsize=9)

fig = plt.figure(figsize=(15, 11), facecolor=BG)
fig.suptitle(
    "BHU Cross-Match · Static Comparison Graph\n"
    "Universe A (Real DESI Data)  vs  Universe B (BHU Super-Eddington Theory)",
    color="#dddddd", fontsize=13, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.42, wspace=0.30,
                       left=0.07, right=0.95,
                       top=0.91,  bottom=0.07)

# Panel 1 — Scale factor (full width)
ax1 = fig.add_subplot(gs[0, :]); style(ax1)
ax1.plot(idx, a_A_hist[idx], color=COLOR_A, lw=2.2, label="Universe A (DESI)")
ax1.plot(idx, a_B_hist[idx], color=COLOR_B, lw=2.2, linestyle="--", label="Universe B (BHU)")
ax1.axhline(1.0, color="#ffffff", lw=0.6, alpha=0.2, linestyle=":")
ax1.set_xlabel("Simulation steps →", color="#888888", fontsize=10)
ax1.set_ylabel("Cosmic Scale Factor  a(t)", color="#888888", fontsize=10)
ax1.set_title("Scale Factor Evolution", color="#cccccc", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 1.3)

ax1d = ax1.twinx(); ax1d.set_facecolor(BG)
ax1d.fill_between(idx, deviation[idx] * 10, alpha=0.12, color=COLOR_D)
ax1d.plot(idx, deviation[idx] * 10, color=COLOR_D, lw=1.2, alpha=0.85, label="Dev ×10")
ax1d.set_ylabel("Deviation % ×10", color=COLOR_D, fontsize=9)
ax1d.tick_params(axis="y", colors=COLOR_D, labelsize=9)
ax1d.set_ylim(-0.5, 10)
for sp in ax1d.spines.values(): sp.set_edgecolor(GRID)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1d.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=9, framealpha=0.3,
           facecolor="#1a1a22", labelcolor="white", edgecolor="#333", loc="upper left")

# Panel 2 — H(z) comparison
ax2 = fig.add_subplot(gs[1, 0]); style(ax2)
ax2.plot(z_range, H_lcdm_c, color=COLOR_L, lw=1.5, linestyle=":", label="ΛCDM baseline")
ax2.plot(z_range, H_emp_c,  color=COLOR_A, lw=2,   label="DESI (interp)")
ax2.plot(z_range, H_theo_c, color=COLOR_B, lw=2,   linestyle="--", label="BHU Theory")
ax2.errorbar(z_desi, H_desi, yerr=H_err, fmt="o", color="#ffffff",
             ecolor="#888888", capsize=3, markersize=5, zorder=5, label="DESI data pts")
ax2.invert_xaxis()
ax2.set_xlabel("Redshift  z", color="#888888", fontsize=10)
ax2.set_ylabel("H(z)  [km/s/Mpc]", color="#888888", fontsize=10)
ax2.set_title("H(z) Model Comparison", color="#cccccc", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8, framealpha=0.3, facecolor="#1a1a22",
           labelcolor="white", edgecolor="#333")

# Panel 3 — Deviation over time
ax3 = fig.add_subplot(gs[1, 1]); style(ax3)
ax3.fill_between(idx, deviation[idx], alpha=0.18, color=COLOR_D)
ax3.plot(idx, deviation[idx], color=COLOR_D, lw=2)
ax3.axhline(0.5, color="#BA7517", lw=0.9, alpha=0.7, linestyle=":", label="0.5% threshold")
ax3.axhline(2.0, color=COLOR_D,   lw=0.9, alpha=0.7, linestyle=":", label="2.0% threshold")
ax3.set_xlabel("Simulation steps →", color="#888888", fontsize=10)
ax3.set_ylabel("Deviation  |aA − aB| / aA  (%)", color="#888888", fontsize=10)
ax3.set_title("Cross-Match Deviation Over Time", color="#cccccc",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=8, framealpha=0.3, facecolor="#1a1a22",
           labelcolor="white", edgecolor="#333")

out_path = f"{OUT}/bhu_comparison_graph.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved  →  {out_path}")
print("Open the 'output/' folder in the file explorer to view.")