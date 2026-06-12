"""
Observable and diagnostic utilities for quantum wave-packet simulations.

These functions are used to check whether the numerical evolution is
physically consistent.
"""

import numpy as np


def probability_density(psi):
    """
    Compute the probability density |psi|^2.
    """

    return np.abs(psi) ** 2


def norm(x, psi):
    """
    Compute the numerical normalisation integral.

    This approximates

        integral |psi(x)|^2 dx

    using a uniform spatial grid.
    """

    dx = x[1] - x[0]

    return np.sum(probability_density(psi)) * dx


def expectation_x(x, psi):
    """
    Compute the expectation value <x>.
    """

    dx = x[1] - x[0]
    density = probability_density(psi)

    return np.sum(x * density) * dx


def width_x(x, psi):
    """
    Compute the spatial width Delta x of the wave packet.
    """

    dx = x[1] - x[0]
    density = probability_density(psi)

    mean_x = np.sum(x * density) * dx
    mean_x2 = np.sum((x**2) * density) * dx

    variance = mean_x2 - mean_x**2

    return np.sqrt(max(variance, 0.0))