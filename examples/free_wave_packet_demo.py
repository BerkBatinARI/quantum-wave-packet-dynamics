"""
free_wave_packet_demo.py

Free-particle Gaussian wave-packet evolution.

This script demonstrates:
1. construction of an initial Gaussian wave packet,
2. unitary time evolution using a finite-difference Hamiltonian,
3. probability-density snapshots,
4. spacetime density heatmap,
5. norm conservation diagnostic,
6. position and width diagnostics.

The output figures are saved in the figures/ directory.
"""

from pathlib import Path
import sys

import numpy as np


# ---------------------------------------------------------------------
# Allow examples/ scripts to import from src/
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.wave_packet import gaussian_wave_packet
from src.potentials import free_potential
from src.solver import build_hamiltonian, evolve_state
from src.observables import (
    probability_density,
    norm,
    expectation_x,
    width_x,
)
from src.visualise import (
    ensure_output_dir,
    plot_wave_packet_snapshots,
    plot_spacetime_density,
    plot_norm_conservation,
    plot_position_diagnostics,
)


def main() -> None:
    """
    Run a free Gaussian wave-packet simulation and save figures.
    """

    # -----------------------------------------------------------------
    # Spatial grid
    # -----------------------------------------------------------------

    x_min = -80.0
    x_max = 80.0
    number_of_points = 1200

    x = np.linspace(x_min, x_max, number_of_points)

    # -----------------------------------------------------------------
    # Initial wave packet
    # -----------------------------------------------------------------

    x0 = -30.0
    sigma = 4.0
    k0 = 1.5

    psi0 = gaussian_wave_packet(
        x=x,
        x0=x0,
        sigma=sigma,
        k0=k0,
    )

    # -----------------------------------------------------------------
    # Potential
    # -----------------------------------------------------------------

    V = free_potential(x)

    # -----------------------------------------------------------------
    # Hamiltonian
    # -----------------------------------------------------------------

    H = build_hamiltonian(
        x=x,
        V=V,
    )

    # -----------------------------------------------------------------
    # Time grid
    # -----------------------------------------------------------------

    t_min = 0.0
    t_max = 35.0
    number_of_times = 240

    times = np.linspace(t_min, t_max, number_of_times)

    # -----------------------------------------------------------------
    # Time evolution
    # -----------------------------------------------------------------

    psi_history = evolve_state(
        psi0=psi0,
        H=H,
        times=times,
    )

    # -----------------------------------------------------------------
    # Observables
    # -----------------------------------------------------------------

    density_history = np.array(
        [probability_density(psi) for psi in psi_history]
    )

    norms = np.array(
        [norm(x, psi) for psi in psi_history]
    )

    expectation_values = np.array(
        [expectation_x(x, psi) for psi in psi_history]
    )

    widths = np.array(
        [width_x(x, psi) for psi in psi_history]
    )

    # -----------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------

    output_dir = ensure_output_dir(PROJECT_ROOT / "figures")

    # -----------------------------------------------------------------
    # Figure 1: snapshots
    # -----------------------------------------------------------------

    snapshot_indices = [0, 60, 120, 180, 239]

    snapshot_densities = [
        density_history[index] for index in snapshot_indices
    ]

    snapshot_times = [
        times[index] for index in snapshot_indices
    ]

    plot_wave_packet_snapshots(
        x=x,
        densities=snapshot_densities,
        times=snapshot_times,
        title="Free Gaussian wave-packet spreading",
        output_path=output_dir / "free_wave_packet_snapshots.png",
    )

    # -----------------------------------------------------------------
    # Figure 2: spacetime heatmap
    # -----------------------------------------------------------------

    plot_spacetime_density(
        x=x,
        times=times,
        density_history=density_history,
        title="Free wave-packet spacetime density",
        output_path=output_dir / "free_wave_packet_spacetime.png",
    )

    # -----------------------------------------------------------------
    # Figure 3: norm conservation
    # -----------------------------------------------------------------

    plot_norm_conservation(
        times=times,
        norms=norms,
        title="Norm conservation for free evolution",
        output_path=output_dir / "free_wave_packet_norm.png",
    )

    # -----------------------------------------------------------------
    # Figure 4: position diagnostics
    # -----------------------------------------------------------------

    plot_position_diagnostics(
        times=times,
        expectation_values=expectation_values,
        widths=widths,
        title="Free wave-packet position diagnostics",
        output_path=output_dir / "free_wave_packet_position_diagnostics.png",
    )

    print("Free wave-packet simulation complete.")
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()