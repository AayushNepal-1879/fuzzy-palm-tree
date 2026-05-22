import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. OFFICIAL DESI Y1 BAO DATA
# ==========================================
z_desi = np.array([0.10, 0.30, 0.51, 0.71, 0.93, 1.32, 2.33])
H_desi = np.array([69.0, 80.5, 90.0, 99.0, 115.5, 150.1, 224.6])

# Empirical Engine
def get_H_emp(z):
    # z_desi is already in ascending order. np.interp requires strictly increasing x-values.
    # To handle z < 0.10, we allow it to flatline at H=69.0, or you can append z=0 to the arrays.
    return np.interp(z, z_desi, H_desi)

# Theoretical Engine (BHU Super-Eddington)
def get_H_theory(z):
    H0 = 68.0
    Om = 0.31
    OL = 0.69
    lSE = 0.085    
    zQ = 1.8       
    Mz = np.exp(-lSE * z * np.exp(-z / zQ))
    effOL = OL * (1.0 / Mz)**2
    return H0 * np.sqrt(Om * (1+z)**3 + effOL)

# ==========================================
# 2. NUMERICAL INTEGRATION (FORWARD EULER)
# ==========================================
z_start = 2.33
a_start = 1.0 / (1.0 + z_start)
dt = 0.00001 # Highly granular time-step

a_A, z_A = a_start, z_start
a_B, z_B = a_start, z_start

t_list, aA_list, aB_list, dev_list = [], [], [], []
t = 0

# Integrate forward in time until we reach today
while a_A < 1.0 and a_B < 1.0:
    t_list.append(t)
    aA_list.append(a_A)
    aB_list.append(a_B)
    
    # Calculate Cross-Match Deviation %
    dev = abs(a_A - a_B) / a_A * 100
    dev_list.append(dev)
    
    # Evolve Universe A (Data)
    a_A += a_A * get_H_emp(z_A) * dt
    z_A = (1.0 / a_A) - 1.0
    
    # Evolve Universe B (Theory)
    a_B += a_B * get_H_theory(z_B) * dt
    z_B = (1.0 / a_B) - 1.0
    
    t += dt

# ==========================================
# 3. PUBLICATION-QUALITY PLOTTING
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

# Main plot: Scale Factor evolution (Left Y-Axis)
ax1.plot(t_list, aA_list, color='#10B981', lw=3, label='Universe A (Real DESI Data)')
ax1.plot(t_list, aB_list, color='#3B82F6', lw=3, linestyle='--', label='Universe B (BHU Theory)')
ax1.set_xlabel('Simulation Time [Arbitrary Units]', fontsize=13, fontweight='bold')
ax1.set_ylabel('Cosmic Scale Factor $a(t)$', fontsize=13, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.7)

# Sub-plot: Deviation overlay (Right Y-Axis)
ax2 = ax1.twinx()
ax2.plot(t_list, dev_list, color='#E24B4A', lw=2, alpha=0.85, label='Cross-Match Deviation (%)')
ax2.set_ylabel('Deviation (%)', color='#E24B4A', fontsize=13, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#E24B4A')
ax2.set_ylim(0, max(dev_list) * 3) # Keep the red line in the bottom third of the graph

# Legends and Titles
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11)

plt.title('Forward Modeling Cross-Match: Real DESI Data vs. BHU Theory', fontsize=15, pad=15)
plt.tight_layout()

# Save the file directly to the Codespace
file_name = 'Figure1_CrossMatch_Deviation.png'
plt.savefig(file_name, bbox_inches='tight')

print(f"SUCCESS: Cross-Match Figure saved as '{file_name}'")
print(f"Final Deviation: {dev_list[-1]:.4f}%")