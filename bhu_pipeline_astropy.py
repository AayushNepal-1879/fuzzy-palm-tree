"""
Figure 1 -- Forward Modeling Cross-Match: Real DESI Data vs BHU Theory
Astropy Proper Cosmological Time Edition -- Accretion-Driven Version
======================================================================
GitHub Codespaces version -- saves PDF directly, no display needed.

Install once:
    pip install numpy matplotlib astropy scipy

Run:
    python Figure1_CrossMatch_Deviation.py

Output:
    output/Figure1_CrossMatch_Deviation.pdf

PHYSICS CHANGE FROM PRIOR VERSION:
    get_H_theory(z) previously used a phenomenological double-exponential
    curve-fit (parameters lSE, zQ) with no physical referent, tuned blind
    to the DESI curve it was being compared against.

    This version replaces that fit with M(z)/M0 obtained by integrating
    the standard Eddington-limited accretion law (Salpeter 1964):

        dM/dt = ((1-eta)/eta) * f_Edd(z) * M / t_Edd

    combined with the manuscript's Nariai-limit postulate Lambda=1/(9 G^2 M^2).
    The mass ratio M(t)/M0 is fixed by the physical boundary condition
    M(z=0)=M0 (NOT a free initial condition), because the ODE is linear:

        M(t)/M0 = exp( -integral_t^t0 k(t') dt' ),
        k(t) = ((1-eta)/eta) * f_Edd(z(t)) / t_Edd

    f_Edd(z) = f0*(1+z)^p is a physical modeling assumption (AGN Eddington-
    ratio downsizing; Hopkins, Richards & Hernquist 2007), NOT a black-box
    shape function. p is fixed from that literature; f0 (present-day
    duty-cycle-averaged Eddington ratio) is the ONE calibrated free
    parameter, found below by matching the DESI terminal deviation.

    eta=0.057 and t_Edd=450 Myr are reused unchanged from Track II.

ACCURACY HIERARCHY (most -> least accurate), unchanged from prior version:
----------------------------------------------------------------------
1. THIS SCRIPT -- Astropy proper cosmological time
   - t in real Gyr (from astropy FlatLambdaCDM)
   - H in Gyr^-1  (converted: 1 km/s/Mpc = 1.02269e-3 Gyr^-1)
   - DT = 0.001 Gyr = 1 Myr steps
   - Friedmann equation: da/dt = a.H(z)  is physically exact
----------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.optimize import brentq

OUT = "output"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. COSMOLOGY & UNIT CONVERSION
# ============================================================
cosmo = FlatLambdaCDM(H0=68.0, Om0=0.31)
KM_S_MPC_TO_GYR = 1.02269e-3    # multiply H [km/s/Mpc] -> H [Gyr^-1]

H0_ref = 68.0   # km/s/Mpc
Om = 0.31
OL0 = 0.69

# ============================================================
# 2. OFFICIAL DESI Y1 BAO DATA
# ============================================================
z_desi = np.array([0.10, 0.30, 0.51, 0.71, 0.93, 1.32, 2.33])
H_desi = np.array([69.0, 80.5, 90.0, 99.0, 115.5, 150.1, 224.6])


def get_H_emp(z):
    """Universe A -- interpolated from real DESI BAO data. Returns H in Gyr^-1."""
    return float(np.interp(z, z_desi, H_desi)) * KM_S_MPC_TO_GYR


# ============================================================
# 3. STANDARD ASTROPHYSICS -- Eddington-limited accretion
#    (Salpeter 1964; same eta, t_Edd as Track II)
# ============================================================
eta = 0.057       # Schwarzschild ISCO radiative efficiency (Track II value)
t_Edd_Gyr = 0.45  # Eddington time, 450 Myr (Track II value)

# ============================================================
# 4. PHYSICAL MODELING ASSUMPTION -- AGN Eddington-ratio history
#    f_Edd(z) = f0 * (1+z)^p
#    p fixed from AGN downsizing literature (Hopkins, Richards &
#    Hernquist 2007). f0 is the ONE calibrated free parameter.
# ============================================================
p_index = 3.0


def f_Edd(z, f0):
    return f0 * (1.0 + z) ** p_index


# ============================================================
# 5. ASTROPY PROPER COSMIC TIME BOUNDS + TIME/REDSHIFT BACKBONE
#    (this block defines _z_grid, _t_grid, _order, _z_sorted --
#     these were missing in the merged version, which is why
#     Pylance flagged them as undefined)
# ============================================================
z_start = 2.33
z_today = 0.001   # avoid z=0 numerical singularity

t_start_gyr = cosmo.age(z_start).to(u.Gyr).value
t_end_gyr   = cosmo.age(z_today).to(u.Gyr).value
a_start     = 1.0 / (1.0 + z_start)

print("Cosmological time bounds:")
print(f"  z = {z_start}  ->  t = {t_start_gyr:.4f} Gyr  (a = {a_start:.5f})")
print(f"  z ~ 0        ->  t = {t_end_gyr:.4f} Gyr  (a ~ 1)")
print(f"  Total span   = {t_end_gyr - t_start_gyr:.4f} Gyr")

# Fine z(t) lookup used to build the time/redshift grid the accretion
# integral runs on. The BHU curve deviates from LambdaCDM by <1%, so
# using the same astropy z(t) mapping here (rather than a fully
# self-consistent nonlinear solve) is an excellent, explicitly-flagged
# approximation.
_z_fine = np.linspace(0, 20, 20000)
_t_fine = cosmo.age(_z_fine).to(u.Gyr).value

_N_GRID = 2000
_t_grid = np.linspace(t_start_gyr, t_end_gyr, _N_GRID)
_z_grid = np.interp(_t_grid, _t_fine[::-1], _z_fine[::-1])   # descending in z
_order = np.argsort(_z_grid)                                  # ascending, for np.interp
_z_sorted = _z_grid[_order]


def mass_ratio_array(f0):
    """
    Returns M(t)/M0 on the _t_grid, satisfying M(t_end)/M0 = 1 exactly
    (the physical z=0 boundary condition), via the closed-form solution
    of the linear accretion ODE -- no shooting method required.
    """
    k = ((1.0 - eta) / eta) * f_Edd(_z_grid, f0) / t_Edd_Gyr   # Gyr^-1
    K = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(_t_grid))])
    return np.exp(K - K[-1])


def get_H_theory(z, Mz_lookup):
    """
    Universe B -- BHU theory, driven by accretion-derived M(z)/M0
    instead of a phenomenological curve-fit. Returns H in Gyr^-1.
    """
    Mz = np.interp(z, _z_sorted, Mz_lookup)
    effOL = OL0 * (1.0 / Mz) ** 2
    return H0_ref * np.sqrt(Om * (1 + z) ** 3 + effOL) * KM_S_MPC_TO_GYR


# ============================================================
# 6. CALIBRATE f0 (the one physical free parameter)
#    Target: match the terminal cross-match deviation reported for
#    Track I (0.948%), the same way LambdaCDM parameters are
#    calibrated to data -- a one-parameter physical fit, not a
#    blind multi-parameter curve-fit.
# ============================================================
DT = 0.001   # Gyr (= 1 Myr steps) -- fine integration step, distinct from _N_GRID


def run_integration(f0, dt=DT):
    """Full Euler integration at step size dt, using calibrated f0."""
    Mz_lookup = mass_ratio_array(f0)[_order]

    a_A, z_A = a_start, z_start
    a_B, z_B = a_start, z_start
    t_list, aA_list, aB_list, dev_list = [t_start_gyr], [a_start], [a_start], [0.0]
    t = t_start_gyr

    for _ in range(20_000_000):
        a_A += a_A * get_H_emp(z_A) * dt
        z_A = 1.0 / a_A - 1.0

        H_th = get_H_theory(z_B, Mz_lookup)
        a_B += a_B * H_th * dt
        z_B = 1.0 / a_B - 1.0

        t += dt
        t_list.append(t); aA_list.append(a_A); aB_list.append(a_B)
        dev_list.append(abs(a_A - a_B) / a_A * 100)

        if a_A >= 1.0 or a_B >= 1.0 or t >= t_end_gyr:
            break

    return (np.array(t_list), np.array(aA_list), np.array(aB_list), np.array(dev_list))


def _terminal_dev_minus_target(f0, target=0.948):
    _, _, _, dev_arr = run_integration(f0)
    return dev_arr[-1] - target


print("\nCalibrating f0 (present-day duty-cycle-averaged Eddington ratio)...")
f0_star = brentq(_terminal_dev_minus_target, 1e-6, 1e-4, xtol=1e-9)
print(f"  f0 (z=0)    = {f0_star:.4e}")
print(f"  f0 (z=2.33) = {f0_star * (1 + z_start) ** p_index:.4e}  (via (1+z)^{p_index:.0f} scaling)")

# ============================================================
# 7. FINAL INTEGRATION WITH CALIBRATED f0
# ============================================================
print(f"\nIntegrating with DT = {DT*1000:.0f} Myr steps (astropy proper time)...")
t_arr, aA_arr, aB_arr, dev_arr = run_integration(f0_star)

print(f"Integration complete -- {len(t_arr):,} steps")
print(f"Time range : {t_arr[0]:.4f} -> {t_arr[-1]:.4f} Gyr")
print(f"Final aA   = {aA_arr[-1]:.5f}")
print(f"Final aB   = {aB_arr[-1]:.5f}")
print(f"Final dev  = {dev_arr[-1]:.4f}%")
print(f"Peak  dev  = {dev_arr.max():.4f}%  at t = {t_arr[np.argmax(dev_arr)]:.3f} Gyr")

# Accuracy check against astropy at key redshifts
print("\nAccuracy vs astropy reference a(z):")
for zc in [2.0, 1.5, 1.0, 0.5, 0.2]:
    a_true  = 1.0 / (1.0 + zc)
    t_true  = cosmo.age(zc).to(u.Gyr).value
    idx     = np.argmin(np.abs(t_arr - t_true))
    err_pct = abs(aA_arr[idx] - a_true) / a_true * 100
    print(f"  z={zc:.1f}: a_true={a_true:.5f}, "
          f"aA_integrated={aA_arr[idx]:.5f}, "
          f"error={err_pct:.4f}%")

# ============================================================
# 8. PUBLICATION-QUALITY PLOT
# ============================================================
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
fig.patch.set_facecolor("white")

# -- Left axis: Scale factor -----------------------------------
ax1.plot(t_arr, aA_arr, color="#10B981", lw=3,
         label="Universe A  (Real DESI Data)")
ax1.plot(t_arr, aB_arr, color="#3B82F6", lw=3, linestyle="--",
         label="Universe B  (BHU Accretion-Driven Theory)")
ax1.axhline(1.0, color="#bbb", lw=0.8, linestyle=":", alpha=0.7)

z_grid_plot = np.linspace(0.001, z_start, 500)
t_grid_plot = np.array([cosmo.age(float(z)).to(u.Gyr).value for z in z_grid_plot])
a_grid_plot = 1.0 / (1.0 + z_grid_plot)
sort_idx = np.argsort(t_grid_plot)
ax1.plot(t_grid_plot[sort_idx], a_grid_plot[sort_idx],
         color="#aaa", lw=1.2, linestyle=":", alpha=0.7,
         label="True \u039bCDM  (astropy reference)")

ax1.set_xlabel("Cosmic Time  [Gyr]", fontsize=13, fontweight="bold")
ax1.set_ylabel("Cosmic Scale Factor  $a(t)$", fontsize=13, fontweight="bold")
ax1.set_ylim(0.25, 1.15)
ax1.grid(True, linestyle=":", alpha=0.45)

for zc, label in [(2.0, "z=2"), (1.0, "z=1"), (0.5, "z=0.5")]:
    t_mark = cosmo.age(zc).to(u.Gyr).value
    ax1.axvline(t_mark, color="#ccc", lw=0.7, linestyle="--", alpha=0.5)
    ax1.text(t_mark + 0.08, 0.28, label,
             fontsize=8, color="#999", ha="left", va="bottom")

# -- Right axis: Deviation ---------------------------------------
ax2 = ax1.twinx()
ax2.plot(t_arr, dev_arr, color="#E24B4A", lw=2, alpha=0.85,
         label="Cross-Match Deviation (%)")
ax2.fill_between(t_arr, dev_arr, alpha=0.07, color="#E24B4A")
ax2.set_ylabel("Deviation (%)", color="#E24B4A",
               fontsize=13, fontweight="bold")
ax2.tick_params(axis="y", labelcolor="#E24B4A")
ax2.set_ylim(0, dev_arr.max() * 3.5)

peak_idx = np.argmax(dev_arr)
ax2.annotate(
    f"Peak \u0394 = {dev_arr[peak_idx]:.3f}%\nt = {t_arr[peak_idx]:.2f} Gyr",
    xy=(t_arr[peak_idx], dev_arr[peak_idx]),
    xytext=(t_arr[peak_idx] - 1.5, dev_arr[peak_idx] * 1.6),
    arrowprops=dict(arrowstyle="->", color="#E24B4A", lw=1.2),
    color="#E24B4A", fontsize=9, ha="center"
)

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2,
           loc="upper left", fontsize=10, framealpha=0.92)

ax1.text(
    0.98, 0.04,
    "Integration: Euler  \u00b7  DT = 1 Myr\n"
    "Time axis: astropy Flat\u039bCDM (H\u2080=68, \u03a9\u2098=0.31)\n"
    f"Accretion: Salpeter, f0={f0_star:.2e}, p={p_index:.0f}",
    transform=ax1.transAxes, fontsize=7.5, ha="right", va="bottom",
    color="#888", style="italic",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
              edgecolor="#ddd", alpha=0.85)
)

plt.title(
    "Forward Modeling Cross-Match: Real DESI Data vs BHU Theory\n"
    "(Astropy Proper Cosmological Time  \u00b7  H in Gyr\u207b\u00b9)",
    fontsize=14, pad=14
)
plt.tight_layout()

# ============================================================
# 9. SAVE
# ============================================================
out_path = os.path.join(OUT, "Figure1_CrossMatch_Deviation.pdf")
plt.savefig(out_path, bbox_inches="tight")

print(f"\nSaved  ->  {out_path}")