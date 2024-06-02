from scipy.optimize import root, OptimizeResult
import numpy as np


def calculate_entropy(mu: np.ndarray, a: np.ndarray) -> float:
    """
    Calculate the entropy of a probability distribution.

    Args:
        mu (np.ndarray): The probability distribution.
        a (np.ndarray): The weights.

    Returns:
        float: The entropy value.
    """
    return -np.sum(
        [0 if mu[i] == 0 else a[i] * mu[i] * np.log(mu[i]) for i in range(len(a))]
    )


def find_root(equation: callable, constraints: callable) -> float:
    """
    Find the root of an equation that satisfies the given constraints.

    Args:
        equation (callable): The equation to solve.
        constraints (callable): The constraints function.

    Returns:
        float: The root value.

    Raises:
        ValueError: If the root value does not satisfy the constraints or if the root finding fails.
    """
    root_result: OptimizeResult = root(func=equation, x0=[0])
    if root_result.success:
        root_value = root_result.x[0]
        if constraints(root_value):
            return root_value
        else:
            raise ValueError("Root value does not satisfy constraints.")
    else:
        raise ValueError("Root finding failed.")


def entropic_sharpening(
    mu: np.ndarray, entropy_bound: float, a: np.ndarray
) -> np.ndarray:
    """
    Perform entropic sharpening on a probability distribution.

    Args:
        mu (np.ndarray): The probability distribution.
        entropy_bound (float): The desired entropy bound.
        a (np.ndarray): The weights.

    Returns:
        np.ndarray: The sharpened probability distribution.

    Notes:
        This function uses the entropic sharpening algorithm to adjust the probability distribution
        to satisfy the desired entropy bound.

    References:
        - Original algorithm: https://arxiv.org/abs/1701.07875
    """
    beta = 1

    if calculate_entropy(mu, a) + np.dot(a.T, mu) > entropy_bound + 1:
        beta = find_root(
            lambda x: np.dot(a.T, mu**x)
            + calculate_entropy(mu**x, a)
            - (1 + entropy_bound),
            lambda x: x >= 0,
        )
    return mu**beta
