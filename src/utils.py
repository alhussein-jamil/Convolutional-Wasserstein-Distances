from PIL import Image
import numpy as np

from typing import List, Optional
from src.wasserstein import wass_bary_2d
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
import plotly.graph_objects as go
import trimesh
from skimage.measure import marching_cubes
from src.mesh import normalize, index_to_xyz as i_to_xyz
import pandas as pd
import plotly.express as px


X_RANGE = Y_RANGE = Z_RANGE = [-0.2, 1.2]  # The range of the plot


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


def distribution_to_binary(distribution: np.ndarray) -> np.ndarray:
    """
    Convert a distribution to a binary array based on a threshold.

    Args:
        distribution (np.ndarray): The distribution array.

    Returns:
        np.ndarray: The binary array.
    """
    return np.where(distribution > distribution.max() / 8, 1, 0)


def smoothed_figure_sub(
    binary: np.ndarray, n: int, plot_id: int
) -> tuple[go.Figure, trimesh.Trimesh]:
    """
    Create a smoothed 3D figure and mesh from a binary array.

    Args:
        binary (np.ndarray): The binary array.
        n (int): The dimension of the binary array.
        plot_id (int): The plot ID.

    Returns:
        go.Figure: The 3D plotly figure.
        trimesh.Trimesh: The smoothed trimesh object.
    """
    # Load binary 3D mesh data
    binary_mesh = binary.reshape((n, n, n))

    # Define the marching cubes parameters
    level = 0.5
    spacing = (1.0, 1.0, 1.0)
    gradient_direction = "descent"

    # Extend binary mesh for better edge detection
    binary_mesh_extended = np.zeros((n + 2, n + 2, n + 2))
    binary_mesh_extended[1 : n + 1, 1 : n + 1, 1 : n + 1] = binary_mesh

    # Create the mesh using marching cubes
    verts, faces, normals, values = marching_cubes(
        binary_mesh_extended,
        level,
        spacing=spacing,
        gradient_direction=gradient_direction,
    )

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    normalize(mesh)

    # Smooth the mesh using the Laplacian smoothing algorithm
    mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=20)

    # Get the new vertices and faces
    new_verts = mesh.vertices
    new_faces = mesh.faces

    # Create a new subplot with the given ID
    subplot = go.Figure(
        go.Mesh3d(
            x=new_verts[:, 0],
            y=new_verts[:, 1],
            z=new_verts[:, 2],
            i=new_faces[:, 0],
            j=new_faces[:, 1],
            k=new_faces[:, 2],
        )
    )

    # Update the layout of the subplot
    subplot.update_layout(
        scene=dict(
            xaxis=dict(range=X_RANGE),
            yaxis=dict(range=Y_RANGE),
            zaxis=dict(range=Z_RANGE),
        ),
        scene_aspectratio=dict(x=1, y=1, z=1),
    )

    return subplot, mesh


def point_cloud_plot(points: np.ndarray) -> None:
    """
    Plot a 3D scatter plot of points.

    Args:
        points (np.ndarray): The array of points to plot.
    """
    if points.size > 0:
        points_df = pd.DataFrame(points, columns=["x", "y", "z"])

        # Plot the points in 3D
        fig = px.scatter_3d(points_df, x="x", y="y", z="z", width=800, height=600)
        fig.show()


def distribution_to_cloud(distribution: np.ndarray, scale: int = 1) -> np.ndarray:
    """
    Convert a distribution to a point cloud.

    Args:
        distribution (np.ndarray): The distribution array.
        scale (int): Scale factor for the point cloud.

    Returns:
        np.ndarray: The point cloud array.
    """
    n = distribution.shape[0]
    points = []
    n_pts = scale * int(1.0 / np.max(distribution) * (1.28747 + 468.153 / n**3))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                n_pts_in_cell = min(1, int(distribution[i, j, k] * n_pts))
                for _ in range(n_pts_in_cell):
                    x = np.random.rand() + i
                    y = np.random.rand() + j
                    z = np.random.rand() + k
                    points.append([x, y, z])
    return np.array(points)


def plot_3d_binary(binary_distribution: np.ndarray, n: int) -> None:
    """
    Plot a 3D binary distribution using marching cubes.

    Args:
        binary_distribution (np.ndarray): The binary distribution array.
        n (int): The dimension of the binary array.
    """
    binary_distribution = binary_distribution.reshape((n, n, n))

    # Create an isosurface with marching cubes algorithm
    vertices, triangles, _, _ = marching_cubes(binary_distribution, level=0)
    vertices = vertices / n

    # Create a 3D scatter plot of the vertices with the triangles
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                color="lightpink",
                opacity=1,
            )
        ]
    )

    # Set the axis range to [0, 1]
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), zaxis=dict(range=[0, 1])
        )
    )

    # Show the plot
    fig.show()


def smoothed_figure(binary: np.ndarray, n: int) -> None:
    """
    Create and show a smoothed 3D figure from a binary array.

    Args:
        binary (np.ndarray): The binary array.
        n (int): The dimension of the binary array.
    """
    # Load binary 3D mesh data
    binary_mesh = binary.reshape((n, n, n))

    # Define the marching cubes parameters
    level = 0.5
    spacing = (1.0, 1.0, 1.0)
    gradient_direction = "descent"

    # Create the mesh using marching cubes
    verts, faces, normals, values = marching_cubes(
        binary_mesh, level, spacing=spacing, gradient_direction=gradient_direction
    )

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    normalize(mesh)

    # Smooth the mesh using the Laplacian smoothing algorithm
    mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=20)

    # Get the new vertices and faces
    new_verts = mesh.vertices
    new_faces = mesh.faces

    # Create a 3D plotly figure
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=new_verts[:, 0],
                y=new_verts[:, 1],
                z=new_verts[:, 2],
                i=new_faces[:, 0],
                j=new_faces[:, 1],
                k=new_faces[:, 2],
            )
        ]
    )

    # Set the axis range to [-0.5, 1.5]
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=X_RANGE),
            yaxis=dict(range=Y_RANGE),
            zaxis=dict(range=Z_RANGE),
        ),
        scene_aspectratio=dict(x=1, y=1, z=1),
    )
    fig.show()


def plot_cubic_mesh(binary_distribution: np.ndarray, n: int) -> None:
    """
    Plot a cubic mesh from a binary distribution.

    Args:
        binary_distribution (np.ndarray): The binary distribution array.
        n (int): The dimension of the binary array.
    """
    x, y, z = np.array([]), np.array([]), np.array([])
    i, j, k = np.array([]), np.array([]), np.array([])
    cnt = 0

    for idx, v in enumerate(binary_distribution):
        if v == 1:
            x_val, y_val, z_val = i_to_xyz(idx, n)
            x = np.concatenate(
                (
                    x,
                    np.array(
                        [
                            x_val,
                            x_val,
                            x_val + 1,
                            x_val + 1,
                            x_val,
                            x_val,
                            x_val + 1,
                            x_val + 1,
                        ]
                    ),
                )
            )
            y = np.concatenate(
                (
                    y,
                    np.array(
                        [
                            y_val,
                            y_val + 1,
                            y_val + 1,
                            y_val,
                            y_val,
                            y_val + 1,
                            y_val + 1,
                            y_val,
                        ]
                    ),
                )
            )
            z = np.concatenate(
                (
                    z,
                    np.array(
                        [
                            z_val,
                            z_val,
                            z_val,
                            z_val,
                            z_val + 1,
                            z_val + 1,
                            z_val + 1,
                            z_val + 1,
                        ]
                    ),
                )
            )

            i = np.concatenate(
                (i, 8 * cnt + np.array([7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]))
            )
            j = np.concatenate(
                (j, 8 * cnt + np.array([3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]))
            )
            k = np.concatenate(
                (k, 8 * cnt + np.array([0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]))
            )
            cnt += 1

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x / n, y=y / n, z=z / n, i=i, j=j, k=k, name="y", showscale=True
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=X_RANGE),
            yaxis=dict(range=Y_RANGE),
            zaxis=dict(range=Z_RANGE),
        ),
        scene_aspectratio=dict(x=1, y=1, z=1),
    )
    fig.show()
