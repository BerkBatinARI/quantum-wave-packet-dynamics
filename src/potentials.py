"""
Potential-energy landscapes for one-dimensional quantum dynamics.

This module defines simple potentials that will be used to study
free evolution, scattering, tunnelling, and structured barriers.
"""

import numpy as np


def free_potential(x):
    """
    Return a zero potential over the spatial grid.

    This represents free-particle evolution with no external potential.
    """

    return np.zeros_like(x)


def square_barrier(x, height, left, right):
    """
    Construct a square potential barrier.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional spatial grid.
    height : float
        Barrier height.
    left : float
        Left edge of the barrier.
    right : float
        Right edge of the barrier.

    Returns
    -------
    V : np.ndarray
        Potential-energy values on the grid.
    """

    V = np.zeros_like(x)
    V[(x >= left) & (x <= right)] = height

    return V