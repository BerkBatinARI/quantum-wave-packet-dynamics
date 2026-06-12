"""
Wave-packet construction utilities.

This module defines functions for constructing initial quantum states
on a one-dimensional spatial grid.
"""

import numpy as np


def gaussian_wave_packet(x, x0, sigma, k0):
    """
    Construct a normalised Gaussian wave packet in one spatial dimension.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional spatial grid.
    x0 : float
        Initial centre of the wave packet.
    sigma : float
        Spatial width of the packet.
    k0 : float
        Central wave number.

    Returns
    -------
    psi : np.ndarray
        Complex-valued wave function evaluated on the grid.

    Notes
    -----
    The wave packet has the form

        psi(x) = A exp[-(x - x0)^2 / (4 sigma^2)] exp(i k0 x)

    where A is chosen so that the continuous wave function is normalised
    approximately on the numerical grid.
    """

    envelope = np.exp(-((x - x0) ** 2) / (4 * sigma**2))
    phase = np.exp(1j * k0 * x)

    psi = envelope * phase

    dx = x[1] - x[0]
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

    return psi / norm