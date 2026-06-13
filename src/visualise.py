"""
visualise.py

Professional plotting utilities for quantum wave-packet simulations.

This module keeps all visual design choices in one place so that example
scripts stay clean and GitHub figures look consistent.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Global plotting style
# ---------------------------------------------------------------------

def set_plot_style() -> None:
    """
    Apply a clean scientific plotting style.

    This avoids unreadable default plots and keeps every figure in the
    project visually consistent.
    """

    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def ensure_output_dir(output_dir: str | Path) -> Path:
    """
    Create the output directory if it does not already exist.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# ---------------------------------------------------------------------
# Plot 1: wave-packet snapshots
# ---------------------------------------------------------------------

def plot_wave_packet_snapshots(
    x: np.ndarray,
    densities: list[np.ndarray],
    times: list[float],
    *,
    potential: np.ndarray | None = None,
    title: str = "Wave-packet time evolution",
    output_path: str | Path | None = None,
) -> None:
    """
    Plot probability density snapshots |psi(x,t)|^2 at selected times.

    Parameters
    ----------
    x:
        Spatial grid.

    densities:
        List of probability density arrays.

    times:
        Times corresponding to each density snapshot.

    potential:
        Optional potential profile V(x). If provided, it is rescaled and
        shown faintly in the background for context.

    title:
        Figure title.

    output_path:
        If given, save the figure to this path.
    """

    set_plot_style()

    fig, ax = plt.subplots()

    for density, time in zip(densities, times):
        ax.plot(x, density, label=fr"$t = {time:.2f}$")

    if potential is not None:
        potential = np.asarray(potential)

        if np.max(np.abs(potential)) > 0:
            density_scale = max(np.max(d) for d in densities)
            potential_scaled = potential / np.max(np.abs(potential))
            potential_scaled = potential_scaled * density_scale * 0.85

            ax.fill_between(
                x,
                0,
                potential_scaled,
                alpha=0.18,
                label=r"scaled $V(x)$",
            )

    ax.set_xlabel(r"Position $x$")
    ax.set_ylabel(r"Probability density $|\psi(x,t)|^2$")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    plt.close(fig)


# ---------------------------------------------------------------------
# Plot 2: spacetime heatmap
# ---------------------------------------------------------------------

def plot_spacetime_density(
    x: np.ndarray,
    times: np.ndarray,
    density_history: np.ndarray,
    *,
    title: str = "Spacetime probability density",
    output_path: str | Path | None = None,
) -> None:
    """
    Plot a heatmap of |psi(x,t)|^2 through time.

    Parameters
    ----------
    x:
        Spatial grid.

    times:
        Time values.

    density_history:
        2D array with shape (number_of_times, number_of_x_points).

    title:
        Figure title.

    output_path:
        If given, save the figure to this path.
    """

    set_plot_style()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    image = ax.imshow(
        density_history,
        extent=[x[0], x[-1], times[-1], times[0]],
        aspect="auto",
        interpolation="nearest",
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(r"$|\psi(x,t)|^2$")

    ax.set_xlabel(r"Position $x$")
    ax.set_ylabel(r"Time $t$")
    ax.set_title(title)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    plt.close(fig)


# ---------------------------------------------------------------------
# Plot 3: norm conservation diagnostic
# ---------------------------------------------------------------------

def plot_norm_conservation(
    times: np.ndarray,
    norms: np.ndarray,
    *,
    title: str = "Norm conservation diagnostic",
    output_path: str | Path | None = None,
) -> None:
    """
    Plot the norm of the wavefunction as a function of time.

    For a well-behaved unitary simulation, this should stay close to 1.
    """

    set_plot_style()

    fig, ax = plt.subplots()

    ax.plot(times, norms)

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=r"ideal norm $=1$",
    )

    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Norm $\int |\psi(x,t)|^2 dx$")
    ax.set_title(title)
    ax.legend(frameon=False)

    norm_min = np.min(norms)
    norm_max = np.max(norms)
    margin = 0.1 * max(abs(norm_max - norm_min), 1e-12)

    ax.set_ylim(norm_min - margin, norm_max + margin)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    plt.close(fig)


# ---------------------------------------------------------------------
# Plot 4: expectation value and width diagnostics
# ---------------------------------------------------------------------

def plot_position_diagnostics(
    times: np.ndarray,
    expectation_values: np.ndarray,
    widths: np.ndarray,
    *,
    title: str = "Position diagnostics",
    output_path: str | Path | None = None,
) -> None:
    """
    Plot <x>(t) and wave-packet width sigma_x(t).

    The two quantities are shown in separate panels because they have
    different physical meanings:
    - <x>(t): centre of the wave packet,
    - sigma_x(t): spatial spread of the wave packet.
    """

    set_plot_style()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 6),
        sharex=True,
    )

    ax_position = axes[0]
    ax_width = axes[1]

    # -------------------------------------------------------------
    # Panel 1: expectation value <x>(t)
    # -------------------------------------------------------------

    ax_position.plot(times, expectation_values)

    ax_position.set_ylabel(r"$\langle x \rangle$")
    ax_position.set_title(title)

    ax_position.text(
        0.02,
        0.88,
        "wave-packet centre",
        transform=ax_position.transAxes,
        fontsize=10,
        alpha=0.75,
    )

    # -------------------------------------------------------------
    # Panel 2: width sigma_x(t)
    # -------------------------------------------------------------

    ax_width.plot(times, widths)

    ax_width.set_xlabel(r"Time $t$")
    ax_width.set_ylabel(r"$\sigma_x$")

    ax_width.text(
        0.02,
        0.88,
        "wave-packet spreading",
        transform=ax_width.transAxes,
        fontsize=10,
        alpha=0.75,
    )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

    plt.close(fig)