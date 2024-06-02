from PIL import Image
import numpy as np

from typing import List, Optional
from src.wasserstein import wass_bary_2d
import matplotlib.pyplot as plt
from pathlib import Path
import imageio


def load_normalized_image(name: str) -> np.ndarray:
    # Load the image
    img = Image.open(name)

    # Convert the image to a binary image (black and white)
    img = img.convert("1")

    # Convert the image to a matrix of ones and zeros
    matrix = np.array(img)
    matrix = np.array(
        [
            [0.0 if matrix[i][j] else 1.0 for j in range(len(matrix[0]))]
            for i in range(len(matrix))
        ]
    )

    return matrix / np.sum(matrix)


def load_BW_portrait(name: str) -> np.ndarray:
    """Load a black and white portrait"""
    # Load the image
    img = Image.open(name)

    # Convert the image to a binary image (black and white)
    img = img.convert("L")

    # Convert the image to a matrix of ones and zeros
    matrix = np.array(img)

    return matrix / np.sum(matrix)  # /np.sum(matrix)


def load_rgb_image(name: str) -> np.ndarray:
    """Load a color (RGB) image"""
    img = Image.open(name)
    matrix = np.array(img, dtype="float")
    A = 255 * np.ones((matrix.shape[0], matrix.shape[0]))
    for k in range(matrix.shape[2]):
        matrix[:, :, k] = A - matrix[:, :, k]
        matrix[:, :, k] /= 255
    return matrix


def interpolate_bw_images(
    mus: List[np.ndarray],
    n_interpolations: int = 10,
    coeff: Optional[List[float]] = None,
    gamma: float = 0.004,
    threshold: float = 1e-5,
    n_iter: int = 100,
    temp_dir: str = None,
) -> List[np.ndarray]:
    """Interpolate between black and white images.

    Args:
        mus (List[np.ndarray]): The list of images.
        n_interpolations (int): The number of interpolations.
        coeff (Optional[List[float]]): The coefficients.
        gamma (float): The regularization parameter.
        threshold (float): The threshold value.
        n_iter (int): The number of iterations.
        temp_dir (str): The temporary directory.

    Returns:
        List[np.ndarray]: The list of images.
    """

    images = []
    n = int(mus[0].shape[1] ** 0.5)

    if n_interpolations == 1 and coeff is not None:
        bary = wass_bary_2d(
            mus=mus,
            coef=coeff,
            a=np.ones((n**2)) / (0.0 + n**2),
            n=n,
            gamma=gamma,
            iterations=n_iter,
            entropic_sharpening=False,
        )
        bary = bary.reshape(n, n)
        if threshold is not None:
            bary = np.where(bary > threshold, 1, 0)
        # Plot the heatmap using imshow
        plt.imshow(bary, cmap="binary")
        plt.colorbar()
        plt.show()
        images.append(bary)
    else:
        if temp_dir is None:
            temp_dir = "temp"
        for i in range(n_interpolations + 1):
            bary = wass_bary_2d(
                mus=mus,
                coef=[i / n_interpolations, (n_interpolations - i) / n_interpolations],
                a=np.ones((n**2)) / (0.0 + n**2),
                n=n,
                gamma=gamma,
                iterations=n_iter,
                entropic_sharpening=False,
            )
            bary = bary.reshape(n, n)
            # Plot the heatmap using imshow
            plt.imshow(bary, cmap="binary")
            # plt.colorbar()
            plt.savefig(Path(temp_dir) / f"monge_kant{i}.png", bbox_inches="tight")
            images.append(imageio.imread(Path(temp_dir) / f"monge_kant{i}.png"))
            plt.close()
    return images
