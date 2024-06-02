import numpy as np


def convolution2D(kernel: np.ndarray, input_array: np.ndarray) -> np.ndarray:
    """
    This function performs a 2D convolution of a kernel with an input array.
    The kernel is assumed to be square. The input array is assumed to be square
    and have the same size as the kernel.
    (Warning: This function expects the gamme to be applied to the kernel before calling it.)

    Args:
    kernel: a 1D numpy array of size n^2
    input_array: a 1D numpy array of size n^2

    Returns:
    a 1D numpy array of size n^2
    """

    n = len(kernel)
    input_array = input_array.reshape(n, n)
    kernel_extended = np.zeros(2 * n - 1)
    kernel_extended[:n] = np.flip(kernel)
    kernel_extended[n - 1 :] = kernel
    kernel = kernel_extended

    conv_rows = []
    for j in range(n):
        conv_rows += [np.convolve(input_array[j, :], kernel).tolist()]
    conv_rows = np.array(conv_rows)
    conv_rows = conv_rows[:, n - 1 : 2 * n - 1]

    conv_columns = []
    for j in range(n):
        conv_columns += [np.convolve(conv_rows[:, j], kernel).tolist()]

    return np.array(conv_columns)[:, n - 1 : 2 * n - 1].T.flatten()


def naive_2d_convolution(
    kernel: np.ndarray, input_array: np.ndarray, gamma: float
) -> np.ndarray:
    """
    This function performs a 2D convolution of a kernel with an input array.
    The kernel is assumed to be square. The input array is assumed to be square
    and have the same size as the kernel.

    Args:
    kernel: a 1D numpy array of size n^2
    input_array: a 1D numpy array of size n^2
    gamma: a float

    Returns:
    a 1D numpy array of size n^2
    """
    n = len(kernel)
    input_array = input_array.reshape(n, n)
    expected_result = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            expected_result[i, j] = sum(
                np.exp(-((i - a) ** 2) / (gamma * n**2))
                * sum(
                    np.exp(-((j - b) ** 2) / (gamma * n**2)) * input_array[a, b]
                    for b in range(n)
                )
                for a in range(n)
            )

    return expected_result.flatten()


def convolution3D(kernel: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Perform a 3D convolution of a kernel with an input array.

    Args:
    kernel: a 1D numpy array of size n
    u: a 1D numpy array of size n^3

    Returns:
    a 1D numpy array of size n^3
    """
    n = len(kernel)
    u = u.reshape(n, n, n)
    kernel_extended = np.zeros(2 * n - 1)
    kernel_extended[:n] = np.flip(kernel)
    kernel_extended[n - 1 :] = kernel
    kernel = kernel_extended

    conv_rows = []
    for depth in range(n):
        temp = []
        for row in range(n):
            temp += [np.convolve(u[depth, row, :], kernel).tolist()]
        conv_rows += [temp]
    conv_rows = np.array(conv_rows)[:, :, n - 1 : 2 * n - 1]

    conv_columns = []
    for depth in range(n):
        temp = []
        for column in range(n):
            temp += [np.convolve(conv_rows[depth, :, column], kernel).tolist()]
        conv_columns += [temp]
    conv_columns = np.array(conv_columns)[:, :, n - 1 : 2 * n - 1]
    for depth in range(n):
        conv_columns[depth, :, :] = conv_columns[depth, :, :].T

    conv_depth = []
    for row in range(n):
        temp = []
        for column in range(n):
            temp += [np.convolve(conv_columns[:, row, column], kernel).tolist()]
        conv_depth += [temp]
    conv_depth = np.transpose(np.array(conv_depth)[:, :, n - 1 : 2 * n - 1], (2, 0, 1))

    return conv_depth.flatten()


def classical3D(kernel: np.ndarray, u: np.ndarray, gamma: float) -> np.ndarray:
    """
    Perform a 3D convolution using classical method.

    Args:
    kernel: a 1D numpy array of size n
    u: a 1D numpy array of size n^3
    gamma: a float

    Returns:
    a 1D numpy array of size n^3
    """
    n = len(kernel)
    v = u.reshape(n, n, n)

    expected_result = np.zeros((n, n, n))
    for depth in range(n):
        for row in range(n):
            for column in range(n):
                expected_result[depth, row, column] = sum(
                    np.exp(-(((depth - r) / n) ** 2) / (gamma / 2))
                    * sum(
                        np.exp(-(((row - a) / n) ** 2) / (gamma / 2))
                        * sum(
                            np.exp(-(((column - b) / n) ** 2) / (gamma / 2))
                            * v[r, a, b]
                            for b in range(n)
                        )
                        for a in range(n)
                    )
                    for r in range(n)
                )

    return expected_result
