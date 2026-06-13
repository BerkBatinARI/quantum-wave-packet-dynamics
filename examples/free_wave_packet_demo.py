import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.wave_packet import gaussian_wave_packet
from src.potentials import free_potential
from src.solver import build_hamiltonian, evolve_state
from src.observables import probability_density, norm, expectation_x, width_x


# ------------------------------------------------------------
# Numerical setup
# ------------------------------------------------------------

x_min = -60.0
x_max = 60.0
n_points = 1200
x = np.linspace(x_min, x_max, n_points)

times = np.linspace(0.0, 8.0, 180)

hbar = 1.0
mass = 1.0

# ------------------------------------------------------------
# Initial wave packet
# ------------------------------------------------------------

x0 = -25.0
sigma = 3.0
k0 = 3.0

psi0 = gaussian_wave_packet(x=x, x0=x0, sigma=sigma, k0=k0)

# ------------------------------------------------------------
# Free-particle Hamiltonian and time evolution
# ------------------------------------------------------------

V = free_potential(x)
H = build_hamiltonian(x=x, V=V, hbar=hbar, mass=mass)

states = evolve_state(psi0=psi0, H=H, times=times, hbar=hbar)

density_time = np.array([probability_density(psi) for psi in states])
norms = np.array([norm(x, psi) for psi in states])
centres = np.array([expectation_x(x, psi) for psi in states])
widths = np.array([width_x(x, psi) for psi in states])

# ------------------------------------------------------------
# Plot style
# ------------------------------------------------------------

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 300,
    }
)

os.makedirs("figures", exist_ok=True)

# ------------------------------------------------------------
# Figure 1: snapshots of wave-packet spreading
# ------------------------------------------------------------

snapshot_indices = [0, 40, 80, 120, 179]

fig, ax = plt.subplots(figsize=(10, 5.5))

for idx in snapshot_indices:
    ax.plot(
        x,
        density_time[idx],
        linewidth=2.0,
        label=rf"$t={times[idx]:.2f}$",
    )

ax.set_title("Free Evolution of a Gaussian Quantum Wave Packet")
ax.set_xlabel(r"Position $x$")
ax.set_ylabel(r"Probability density $|\psi(x,t)|^2$")
ax.legend(frameon=False)
ax.grid(alpha=0.18)

fig.tight_layout()
fig.savefig("figures/free_wave_packet_snapshots.png")
plt.show()

# ------------------------------------------------------------
# Figure 2: spacetime density map
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

image = ax.imshow(
    density_time,
    extent=[x_min, x_max, times[-1], times[0]],
    aspect="auto",
    interpolation="bilinear",
)

cbar = fig.colorbar(image, ax=ax)
cbar.set_label(r"Probability density $|\psi(x,t)|^2$")

ax.set_title("Spacetime Evolution of the Probability Density")
ax.set_xlabel(r"Position $x$")
ax.set_ylabel(r"Time $t$")

fig.tight_layout()
fig.savefig("figures/free_wave_packet_spacetime.png")
plt.show()

# ------------------------------------------------------------
# Figure 3: diagnostics
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(times, norms, linewidth=2.2, label=r"Normalisation")
ax.set_title("Numerical Consistency Check")
ax.set_xlabel(r"Time $t$")
ax.set_ylabel(r"$\int |\psi(x,t)|^2 dx$")
ax.grid(alpha=0.18)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig("figures/free_wave_packet_norm.png")
plt.show()