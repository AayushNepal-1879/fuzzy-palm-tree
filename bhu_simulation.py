"""
BHU Cross-Match Simulation — GitHub Codespaces Version
=======================================================
Saves outputs directly to the output/ folder:
  output/bhu_animation.gif          animated simulation (~120 frames)
  output/bhu_universe_A_final.png   Universe A final snapshot
  output/bhu_universe_B_final.png   Universe B final snapshot

Run:
    python bhu_simulation.py

Progress printed to terminal. Click any file in the
Codespace file explorer (left sidebar) to preview it.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # headless — no GUI in Codespaces
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# ── Output folder ─────────────────────────────────────────
OUT = "output"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1. REAL DESI / BAO 2024 OBSERVATIONAL DATA
# ─────────────────────────────────────────────────────────
z_desi = np.array([0.10, 0.30, 0.51, 0.71, 0.93, 1.32, 2.33])
H_desi = np.array([69.0, 80.5, 90.0, 99.0, 115.5, 150.1, 224.6])

def get_H_empirical(z):
    return float(np.interp(z, z_desi, H_desi))

# ─────────────────────────────────────────────────────────
# 2. BHU SUPER-EDDINGTON THEORETICAL ENGINE
# ─────────────────────────────────────────────────────────
H0, Om, OL, lSE, zQ = 68.0, 0.31, 0.69, 0.085, 1.8

def get_H_theory(z):
    Mz     = np.exp(-lSE * z * np.exp(-z / zQ))
    eff_OL = OL * (1.0 / Mz) ** 2
    return H0 * np.sqrt(Om * (1 + z)**3 + eff_OL)

# ─────────────────────────────────────────────────────────
# 3. NUMERICAL INTEGRATION
#    H is normalised by H0 so it's dimensionless (~1 to 3).
#    DT=0.0001 gives ~7,600 smooth integration steps.
# ─────────────────────────────────────────────────────────
z_start = 2.33
a_start = 1.0 / (1.0 + z_start)   # ≈ 0.3003
DT      = 0.0001                   # normalised timestep

print("Integrating physics...")
a_A_hist, a_B_hist = [a_start], [a_start]
z_A, z_B = z_start, z_start
a_A, a_B = a_start, a_start

for _ in range(10_000_000):
    # Normalise H by H0 so DT is in dimensionless Hubble-time units
    a_A += a_A * (get_H_empirical(z_A) / H0) * DT
    z_A  = 1.0 / a_A - 1.0
    a_B += a_B * (get_H_theory(z_B) / H0) * DT
    z_B  = 1.0 / a_B - 1.0
    a_A_hist.append(a_A)
    a_B_hist.append(a_B)
    if a_A >= 1.0 or a_B >= 1.0:
        break

a_A_hist    = np.array(a_A_hist)
a_B_hist    = np.array(a_B_hist)
total_steps = len(a_A_hist)
print(f"Integration complete — {total_steps:,} steps")

# 120 evenly-spaced frame indices across the full run
N_FRAMES      = 120
frame_indices = np.linspace(0, total_steps - 1, N_FRAMES, dtype=int)

# ─────────────────────────────────────────────────────────
# 4. DETERMINISTIC GALAXY PARTICLES (fixed seed)
# ─────────────────────────────────────────────────────────
rng   = np.random.default_rng(seed=0xBEEF1234)
n_p   = 120
theta = rng.uniform(0, 2 * np.pi, n_p)
r_raw = np.sqrt(rng.uniform(0, 1, n_p))
px    = r_raw * np.cos(theta)
py    = r_raw * np.sin(theta)

# ─────────────────────────────────────────────────────────
# 5. FIGURE LAYOUT
# ─────────────────────────────────────────────────────────
COLOR_A = "#10B981"
COLOR_B = "#378ADD"
COLOR_D = "#E24B4A"
BG      = "#0d0d0f"
GRID    = "#2a2a35"

fig = plt.figure(figsize=(14, 9), facecolor=BG)
fig.suptitle(
    "BHU Cross-Match  ·  Empirical Reality (DESI) vs Theoretical Physics (BHU)",
    color="#dddddd", fontsize=13, fontweight="bold", y=0.97
)

gs = gridspec.GridSpec(
    2, 3, figure=fig,
    height_ratios=[2.2, 1.4],
    hspace=0.38, wspace=0.30,
    left=0.05, right=0.97,
    top=0.91,  bottom=0.08
)

ax_A   = fig.add_subplot(gs[0, 0])
ax_B   = fig.add_subplot(gs[0, 1])
ax_tel = fig.add_subplot(gs[0, 2])
ax_g   = fig.add_subplot(gs[1, :])

for ax in [ax_A, ax_B, ax_tel, ax_g]:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

# Universe canvases
for ax, label, color in [(ax_A, "Universe A · DESI Data", COLOR_A),
                         (ax_B, "Universe B · BHU Theory", COLOR_B)]:
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, color=color, fontsize=11, fontweight="bold", pad=6)

scat_A = ax_A.scatter([], [], s=14, c=COLOR_A, alpha=0.75, linewidths=0)
scat_B = ax_B.scatter([], [], s=14, c=COLOR_B, alpha=0.75, linewidths=0)
ring_A = Circle((0, 0), a_start, fill=False, color=COLOR_A, lw=0.8, alpha=0.35)
ring_B = Circle((0, 0), a_start, fill=False, color=COLOR_B, lw=0.8, alpha=0.35)
ax_A.add_patch(ring_A)
ax_B.add_patch(ring_B)

# Telemetry panel
ax_tel.set_xlim(0, 1); ax_tel.set_ylim(0, 1)
ax_tel.set_xticks([]); ax_tel.set_yticks([])
ax_tel.set_title("Live Telemetry", color="#cccccc", fontsize=11, fontweight="bold", pad=6)

tel_labels = ["Scale A (DESI)", "Scale B (BHU)", "Deviation"]
tel_y      = [0.72, 0.50, 0.28]
tel_colors = [COLOR_A, COLOR_B, COLOR_D]

for lbl, y in zip(tel_labels, tel_y):
    ax_tel.text(0.08, y + 0.10, lbl, color="#888888", fontsize=9,
                transform=ax_tel.transAxes, va="bottom")

tel_vals = [
    ax_tel.text(0.08, y, "—", color=c, fontsize=16, fontweight="bold",
                transform=ax_tel.transAxes, va="bottom")
    for y, c in zip(tel_y, tel_colors)
]

# Graph
ax_g.set_xlim(0, total_steps); ax_g.set_ylim(0.0, 1.35)
ax_g.set_xlabel("Simulation steps →", color="#888888", fontsize=10)
ax_g.set_ylabel("Cosmic Scale Factor  a(t)", color="#888888", fontsize=10)
ax_g.set_title("Scale Factor Evolution  ·  Universe A vs B",
               color="#cccccc", fontsize=11, fontweight="bold")
ax_g.tick_params(colors="#666666", labelsize=9)

# Faint full-run background guide lines
idx_bg = np.linspace(0, total_steps - 1, 2000, dtype=int)
ax_g.plot(idx_bg, a_A_hist[idx_bg], color=COLOR_A, lw=0.7, alpha=0.15)
ax_g.plot(idx_bg, a_B_hist[idx_bg], color=COLOR_B, lw=0.7, alpha=0.15, linestyle="--")

line_A, = ax_g.plot([], [], color=COLOR_A, lw=2.0)
line_B, = ax_g.plot([], [], color=COLOR_B, lw=2.0, linestyle="--")

ax_d = ax_g.twinx()
ax_d.set_facecolor(BG); ax_d.set_ylim(-0.5, 10)
ax_d.set_ylabel("Deviation % ×10", color=COLOR_D, fontsize=9)
ax_d.tick_params(axis="y", colors=COLOR_D, labelsize=9)
for sp in ax_d.spines.values(): sp.set_edgecolor(GRID)
line_D, = ax_d.plot([], [], color=COLOR_D, lw=1.2, alpha=0.8)

ax_g.legend(handles=[
    mpatches.Patch(color=COLOR_A, label="Universe A (DESI)"),
    mpatches.Patch(color=COLOR_B, label="Universe B (BHU)"),
    mpatches.Patch(color=COLOR_D, label="Deviation ×10"),
], loc="upper left", fontsize=9, framealpha=0.3,
   facecolor="#1a1a22", labelcolor="white", edgecolor="#333")

# ─────────────────────────────────────────────────────────
# 6. ANIMATION UPDATE FUNCTION
# ─────────────────────────────────────────────────────────
def update(frame_num):
    idx = frame_indices[frame_num]
    aA  = a_A_hist[idx]
    aB  = a_B_hist[idx]

    # Universe canvases
    scat_A.set_offsets(np.c_[px * aA, py * aA])
    scat_B.set_offsets(np.c_[px * aB, py * aB])
    ring_A.set_radius(min(aA, 1.0))
    ring_B.set_radius(min(aB, 1.0))

    # Graph lines (subsample to keep rendering fast)
    xs = np.linspace(0, idx, min(idx + 1, 600), dtype=int)
    line_A.set_data(xs, a_A_hist[xs])
    line_B.set_data(xs, a_B_hist[xs])
    dev = np.abs(a_A_hist[xs] - a_B_hist[xs]) / a_A_hist[xs] * 100
    line_D.set_data(xs, dev * 10)

    # Telemetry
    deviation = abs(aA - aB) / aA * 100
    tel_vals[0].set_text(f"{aA:.5f}")
    tel_vals[1].set_text(f"{aB:.5f}")
    tel_vals[2].set_text(f"{deviation:.4f}%")
    tel_vals[2].set_color(
        COLOR_A if deviation < 0.5 else ("#BA7517" if deviation < 2 else COLOR_D)
    )

    if frame_num % 20 == 0:
        pct = int((frame_num + 1) / N_FRAMES * 100)
        print(f"  Frame {frame_num + 1:>3}/{N_FRAMES}  ({pct}%)", flush=True)

    return scat_A, scat_B, ring_A, ring_B, line_A, line_B, line_D, *tel_vals

# ─────────────────────────────────────────────────────────
# 7. SAVE GIF
#    Keep reference to `anim` until after save() — otherwise
#    Python garbage-collects it before rendering finishes.
# ─────────────────────────────────────────────────────────
gif_path = f"{OUT}/bhu_animation.gif"
print(f"\nRendering {N_FRAMES} frames → {gif_path} ...")

anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=80, blit=False)
anim.save(gif_path, writer=PillowWriter(fps=12), dpi=100)

print(f"Saved  →  {gif_path}")
plt.close(fig)

# ─────────────────────────────────────────────────────────
# 8. FINAL STATE SNAPSHOTS
# ─────────────────────────────────────────────────────────
def save_snapshot(title, a_val, color, filename):
    fig2, ax2 = plt.subplots(figsize=(5, 5), facecolor=BG)
    ax2.set_facecolor(BG)
    ax2.set_xlim(-1.15, 1.15); ax2.set_ylim(-1.15, 1.15)
    ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values(): sp.set_edgecolor(GRID)
    ax2.set_title(f"{title}  ·  a = {a_val:.4f}",
                  color=color, fontsize=11, fontweight="bold")
    ax2.scatter(px * a_val, py * a_val, s=16, c=color, alpha=0.8, linewidths=0)
    ax2.add_patch(Circle((0, 0), min(a_val, 1.0),
                         fill=False, color=color, lw=1, alpha=0.5))
    fig2.savefig(filename, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig2)
    print(f"Saved  →  {filename}")

save_snapshot("Universe A · DESI Data",  float(a_A_hist[-1]), COLOR_A,
              f"{OUT}/bhu_universe_A_final.png")
save_snapshot("Universe B · BHU Theory", float(a_B_hist[-1]), COLOR_B,
              f"{OUT}/bhu_universe_B_final.png")

print("\nAll done! Open the 'output/' folder in the file explorer to view.")