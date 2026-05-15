import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

OUT = "output"
os.makedirs(OUT, exist_ok=True)

# ==========================================
# 1. COSMOLOGY & CONSTANTS
# ==========================================
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)

t_Edd = 450e6      # 450 Myr (ORTE Eq 12 scaling)
eta_0 = 0.057      # Schwarzschild ISCO efficiency
f_B = 0.3          # Magnetic flux compression 

z_seed = 15.0
M_seed = 1e4       # DCBH Seed
t_seed = cosmo.age(z_seed).to(u.yr).value

M_gas_initial = 4e9  
mdot_peak = 800.0  # The PTMAD Super-Eddington burst rate

# ==========================================
# 2. EPISODIC ACCRETION ODE
# ==========================================
def episodic_system(t, y):
    M_bh, M_gas = y
    
    # Gas starvation fraction
    gas_fraction = max(0.0, M_gas / M_gas_initial)
    
    # PHYSICAL FIX: Duty cycle derived from ORTE Sec 2.1.4
    # Burst duration (~100 yr) / Dynamical fallback time (~30 Myr)
    duty_cycle = 100.0 / 30e6  
    
    # The true time-averaged accretion rate
    mdot_effective = mdot_peak * duty_cycle * gas_fraction
    
    # Radiative efficiency is dictated by the physics of the PEAK burst state
    r_tr = max(1.0, 6.75 * mdot_peak) 
    eta_ptmad = eta_0 * (np.log(r_tr) / (2.0 * mdot_peak)) * (1.0 + f_B)
    eta_ptmad = float(np.clip(eta_ptmad, 1e-4, eta_0))
    
    # Mass growth using time-averaged effective rate. 
    # The artificial clamp is removed.
    dM_dt = ((1.0 - eta_ptmad) / eta_ptmad) * (mdot_effective * M_bh / t_Edd)
    
    # Gas blown out by the choked wind (ORTE Eq 15 logic)
    mass_loading = 18.0 
    dM_gas_dt = -dM_dt * (1.0 + mass_loading) if M_gas > 1e3 else 0.0
    
    return [dM_dt, dM_gas_dt]

def standard_growth(t, y):
    # Standard sub-Eddington growth with 10% duty cycle
    return [(y[0] / 45e6) * 0.1, 0.0]

# ==========================================
# 3. NUMERICAL INTEGRATION
# ==========================================
t_end = cosmo.age(4.0).to(u.yr).value
t_eval = np.linspace(t_seed, t_end, 2000)

sol_epi = solve_ivp(episodic_system, [t_seed, t_end], [M_seed, M_gas_initial], t_eval=t_eval, method="LSODA")
sol_std = solve_ivp(standard_growth, [t_seed, t_end], [M_seed, M_gas_initial], t_eval=t_eval, method="LSODA")

M_epi = sol_epi.y[0]
M_std = sol_std.y[0]

# Time to Redshift mapping (ascending for np.interp)
z_lookup = np.linspace(2, 20, 2000)
t_lookup = cosmo.age(z_lookup).to(u.yr).value
t_asc = t_lookup[::-1]
z_asc = z_lookup[::-1]

z_epi = np.interp(sol_epi.t, t_asc, z_asc)
z_std = np.interp(sol_std.t, t_asc, z_asc)

# ==========================================
# 4. JWST EMPIRICAL DATA
# ==========================================
jwst_z = np.array([4.5, 6.5, 8.5, 10.0])
jwst_M = np.array([5e8, 3e7, 2e6, 8e5])  
jwst_err = np.array([2e8, 1e7, 1e6, 4e5])

# ==========================================
# 5. PLOTTING
# ==========================================
fig, ax = plt.subplots(figsize=(10, 5.625), dpi=300)
fig.patch.set_facecolor("white")

ax.plot(z_std, M_std, color="gray", linestyle=":", lw=3, label="Standard Eddington Accretion")
ax.plot(z_epi, M_epi, color="#185FA5", lw=3, label="Episodic Super-Eddington Growth (BHU)")

ax.errorbar(jwst_z, jwst_M, yerr=jwst_err, fmt="o", color="#E24B4A", markersize=7, capsize=4, zorder=5, label="JWST Confirmed LRDs")

ax.set_yscale("log")
ax.set_xlim(16, 3)
ax.set_ylim(5e3, 5e9)
ax.set_xlabel("Redshift $z$", fontsize=14, fontweight="bold")
ax.set_ylabel("Black Hole Mass [$M_\odot$]", fontsize=14, fontweight="bold")
ax.set_title("Early Universe SMBH Growth: Episodic Accretion Model", fontsize=15, pad=15)
ax.grid(True, linestyle="--", alpha=0.5, which="both")
ax.legend(fontsize=12, loc="lower left")

plt.tight_layout()
out_path = os.path.join(OUT, "Figure2_Episodic_Growth.png")
plt.savefig(out_path, bbox_inches="tight", dpi=300)
plt.close(fig)

print("=" * 50)
print(f"SUCCESS: Physical Episodic curve generated -> {out_path}")
print(f"Final BH Mass: {M_epi[-1]:.2e} M_sun")
print("=" * 50)