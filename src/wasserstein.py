import numpy as np
from .convolution import convolution2D, convolution3D
from .post_processing import (
    entropic_sharpening as entropic_sharpening_func,
    calculate_entropy,
)
from typing import List, Union, Optional

from scipy import linalg as slin


def wass_bary_2d(
    mus: List[np.ndarray],
    coef: List[Union[int, float]],
    a: np.ndarray,
    n: int,
    gamma: float = 0.01,
    iterations: int = 100,
    entropic_sharpening: bool = True,
) -> np.ndarray:
    """
    Compute the Wasserstein barycenter of a set of probability distributions.

    Args:
        mus (List[np.ndarray]): The list of probability distributions.
        coef (List[Union[int, float]]): The list of coefficients.
        a (np.ndarray): The weights.
        n (int): The size of the distributions.
        gamma (float): The regularization parameter.
        iterations (int): The number of iterations.
        entropic_sharpening (bool): Whether to use entropic sharpening.

    Returns:
        np.ndarray: The Wasserstein barycenter.
    """

    k = len(mus)
    v = np.ones((k, n**2))
    w = np.ones((k, n**2))
    d = np.zeros((k, n**2))
    eps = 1e-20
    for i in range(k):
        mus[i] += eps
    E = np.exp([-i * i / (gamma / 2) / n**2 for i in range(n)])
    for _ in range(iterations):
        bary = np.ones(n**2)
        for i in range(k):
            w[i] = mus[i] / convolution2D(E, a * v[i])
            d[i] = v[i] * convolution2D(E, a * w[i])
            bary = bary * (d[i] ** coef[i])
        if entropic_sharpening:
            bary = entropic_sharpening_func(
                bary, max(calculate_entropy(u, np.array(a)) for u in mus), np.array(a)
            )
        for i in range(k):
            v[i] = v[i] * bary / d[i]

    return bary


def wass_bary_3d(
    mus: List[np.ndarray],
    coef: List[Union[int, float]],
    a: np.ndarray,
    n: int,
    L: Optional[np.ndarray] = None,
    prefactorized: bool = True,
    conv: bool = False,
    iterations: int = 100,
    gamma: float = 0.01,
    entropic_sharpening: bool = True,
) -> np.ndarray:
    """
    Compute the Wasserstein barycenter of a set of probability distributions in 3D.

    Args:
        mus (List[np.ndarray]): The list of probability distributions.
        coef (List[Union[int, float]]): The list of coefficients.
        a (np.ndarray): The weights.
        n (int): The size of the distributions.
        L (Optional[np.ndarray]): The factorization matrix L.
        prefactorized (bool): Whether the factorization matrix L is pre-factorized.
        conv (bool): Whether to use convolution.
        iterations (int): The number of iterations.
        gamma (float): The regularization parameter.
        entropic_sharpening (bool): Whether to use entropic sharpening.

    Returns:
        np.ndarray: The Wasserstein barycenter.
    """
    k = len(mus)
    v = np.ones((k, n**3))
    w = np.ones((k, n**3))
    d = np.zeros((k, n**3))
    eps = 1e-20
    E = np.exp([-i * i / (gamma / 2) / n**2 for i in range(n)])

    # Add small epsilon to the distributions
    for i in range(k):
        mus[i] += eps

    if not prefactorized and not conv:
        L_inv = np.linalg.inv(L)

    for _ in range(iterations):
        bary = np.ones(n**3)
        for i in range(k):
            if not conv:
                if prefactorized:
                    w[i] = mus[i] / slin.solve_triangular(
                        L.T, slin.solve_triangular(L, a * v[i], lower=True)
                    )
                    d[i] = v[i] * slin.solve_triangular(
                        L.T, slin.solve_triangular(L, a * w[i], lower=True)
                    )
                else:
                    w[i] = mus[i] / (L_inv @ (a * v[i]))
                    d[i] = v[i] * (L_inv @ (a * w[i]))
            else:
                w[i] = mus[i] / convolution3D(E, a * v[i])
                d[i] = v[i] * convolution3D(E, a * w[i])
            bary = bary * (d[i] ** coef[i])

        if entropic_sharpening:
            bary = entropic_sharpening_func(
                bary, max(calculate_entropy(u, np.array(a)) for u in mus), np.array(a)
            )

        for i in range(k):
            v[i] = v[i] * (bary / d[i])

    return bary
