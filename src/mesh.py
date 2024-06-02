import numpy as np
import trimesh
from heapq import heappush, heappop


def geodesic_distance(
    point1: np.ndarray, point2: np.ndarray, mesh: trimesh.Trimesh
) -> float:
    """
    Compute the geodesic distance between two points on a mesh.

    Args:
        point1 (np.ndarray): The first point.
        point2 (np.ndarray): The second point.

    Returns:
        float: The geodesic distance between the two points.
    """

    # Compute the mesh connectivity
    adjacency = mesh.vertex_neighbors

    # Initialize the distances and visited flags
    distances = np.full(len(mesh.vertices), np.inf)
    visited = np.full(len(mesh.vertices), False)

    # Define a priority queue for Dijkstra's algorithm
    queue = []
    heappush(queue, (0, np.where(np.all(mesh.vertices == point1, axis=1))[0][0]))

    # Run Dijkstra's algorithm
    while len(queue) > 0:
        dist, vertex = heappop(queue)
        if visited[vertex]:
            continue
        visited[vertex] = True
        distances[vertex] = dist
        if np.all(mesh.vertices[vertex] == point2):
            break
        for neighbor in adjacency[vertex]:
            if not visited[neighbor]:
                edge_length = np.linalg.norm(
                    mesh.vertices[neighbor] - mesh.vertices[vertex]
                )
                heappush(queue, (dist + edge_length, neighbor))

    # Compute the geodesic distance
    g_dis = distances[np.where(np.all(mesh.vertices == point2, axis=1))[0][0]]

    return g_dis


def index_to_xyz(index: int, n: int) -> tuple[int, int, int]:
    """Unflattens a 1D array index to 3D coordinates (x, y, z)."""
    z = index // (n * n)
    index %= n * n
    y = index // n
    x = index % n
    return x, y, z


def xyz_to_index(x: int, y: int, z: int, n: int) -> int:
    """Flattens 3D coordinates (x, y, z) to a 1D array index."""
    return n * n * x + n * y + z


def write_off(filename: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Writes mesh to an .off file."""
    with open(filename, "w") as file:
        file.write("OFF\n")
        file.write(f"{len(vertices)} {len(faces)} 0\n")
        for vert in vertices:
            file.write(f"{vert[0]} {vert[1]} {vert[2]}\n")
        for face in faces:
            file.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def normalize(mesh: trimesh.Trimesh) -> None:
    """
    Normalize the mesh to fit within the unit cube [0,1]^3.

    Args:
        mesh (trimesh.Trimesh): The mesh to be normalized.
    """
    # Find the scale factor to fit the bounding box in [0,1]^3
    scale_factor = min(0.9 / (mesh.bounds.max(axis=0) - mesh.vertices.min(axis=0)))

    # Apply the scale factor to the mesh
    mesh.apply_scale(scale_factor)

    # Translate the mesh so that its minimum vertex is at the origin
    mesh.apply_translation(-mesh.vertices.min(axis=0))
    min_x, min_y, min_z = mesh.vertices.min(axis=0)
    max_x, max_y, max_z = mesh.vertices.max(axis=0)
    mesh.apply_translation(
        [
            0.5 - (min_x + max_x) / 2,
            0.5 - (min_y + max_y) / 2,
            0.5 - (min_z + max_z) / 2,
        ]
    )


class Mesh:
    mesh: trimesh.base.Trimesh
    n: int
    distribution: np.ndarray
    binary_distribution: np.ndarray
    binary: np.ndarray
    count: np.ndarray
    points: list[np.ndarray]
    cot_laplacian: np.ndarray

    def __init__(self, n: int = 40):
        self.n = n
        self.distribution = None
        self.binary_distribution = None
        self.binary = None
        self.count = None
        self.mesh = None
        self.points = []
        self.cot_laplacian = None

    def update_distribution(self, num_points: int) -> None:
        """Adds more points to the Monte Carlo method."""
        points = np.random.rand(num_points, 3)
        mask = self.mesh.contains(points)
        for point, is_inside in zip(points, mask):
            if is_inside:
                x, y, z = (int(coord * self.n) for coord in point)
                self.points.append(point)
                self.count[xyz_to_index(x, y, z, self.n)] += 1
        self.distribution = self.count / np.sum(self.count)
        self.binary[self.count > 0] = 1
        self.binary_distribution = self.binary / np.sum(self.binary)

    def find_distribution(self, n_cuts: int = 20, num_points: int = 10000) -> None:
        """Creates a distribution on [0,1]^3 after being cut into n_cuts^3 elements."""
        self.n = n_cuts
        points = np.random.rand(num_points, 3)
        self.binary_distribution = np.zeros(n_cuts**3)
        self.binary = np.zeros(n_cuts**3)
        mask = self.mesh.contains(points)
        print("Done masking")
        self.distribution = np.zeros(n_cuts**3)
        inside_points = []

        for point, is_inside in zip(points, mask):
            if is_inside:
                x, y, z = (int(coord * n_cuts) for coord in point)
                inside_points.append(point)
                self.distribution[xyz_to_index(x, y, z, n_cuts)] += 1

        self.count = self.distribution.copy()
        self.distribution /= np.sum(self.distribution)
        print("Done calculating distribution")
        self.binary[self.count > 0] = 1
        self.binary_distribution = self.binary / np.sum(self.binary)
        self.points = inside_points

    def fill_bin(self) -> None:
        """Fills empty spaces in the 0-1 cubical mesh."""
        binary = self.binary.copy().reshape((self.n, self.n, self.n))
        for i in range(1, self.n - 1):
            for j in range(1, self.n - 1):
                for k in range(1, self.n - 1):
                    if binary[i, j, k] == 0:
                        neighbors = [
                            binary[i, j, k + 1],
                            binary[i, j, k - 1],
                            binary[i, j + 1, k],
                            binary[i, j - 1, k],
                            binary[i + 1, j, k],
                            binary[i - 1, j, k],
                            binary[i + 1, j + 1, k + 1],
                            binary[i - 1, j - 1, k - 1],
                            binary[i + 1, j + 1, k - 1],
                            binary[i - 1, j - 1, k + 1],
                            binary[i + 1, j - 1, k + 1],
                            binary[i - 1, j + 1, k - 1],
                            binary[i + 1, j - 1, k - 1],
                            binary[i - 1, j + 1, k + 1],
                        ]
                        if any(neighbors):
                            binary[i, j, k] = 1
        self.binary = binary.flatten()
        self.binary_distribution = self.binary / np.sum(self.binary)

    def normalize_mesh(self) -> None:
        """Normalizes the mesh to fit within the unit cube [0, 1]^3."""
        scale_factor = min(
            0.90 / (self.mesh.bounds.max(axis=0) - self.mesh.vertices.min(axis=0))
        )
        self.mesh.apply_scale(scale_factor)
        self.mesh.apply_translation(-self.mesh.vertices.min(axis=0))
        bounds_min = self.mesh.vertices.min(axis=0)
        bounds_max = self.mesh.vertices.max(axis=0)
        translation = [0.5 - (bounds_min[i] + bounds_max[i]) / 2 for i in range(3)]
        self.mesh.apply_translation(translation)

    def cotangent_laplacian_with_area(self):
        cot_weights = {}
        for face in self.mesh.faces:
            i, j, k = face
            v1 = self.mesh.vertices[i]
            v2 = self.mesh.vertices[j]
            v3 = self.mesh.vertices[k]
            a = np.linalg.norm(v2 - v3)
            b = np.linalg.norm(v1 - v3)
            c = np.linalg.norm(v1 - v2)
            s = (a + b + c) / 2
            # area weight
            A = np.sqrt(s * (s - a) * (s - b) * (s - c))

            # angles of a triangle
            alpha = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
            beta = np.arccos((a**2 + c**2 - b**2) / (2 * a * c))
            gamma = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

            # their cot
            cot_alpha = A / np.sin(alpha)
            cot_beta = A / np.sin(beta)
            cot_gamma = A / np.sin(gamma)

            cot_weights[(i, j)] = cot_alpha
            cot_weights[(j, k)] = cot_beta
            cot_weights[(k, i)] = cot_gamma
        n = len(self.mesh.vertices)
        laplacian = np.zeros((n, n))
        for (i, j), w in cot_weights.items():
            laplacian[i, i] += w
            laplacian[j, j] += w
            laplacian[i, j] -= w
        self.laplacian = laplacian
        self.cot_laplacian = laplacian

    def gaussain_on_mesh(self, vertex, sigma=0.1):
        vertices = self.mesh.vertices
        center = vertices[vertex]
        geodesic_distances = np.array(
            [geodesic_distance(center, v, self.mesh) for v in vertices]
        )
        # define the Gaussian distribution function
        A = 1.0  # scaling factor

        # standard deviation of the distribution
        def gaussian_func(x):
            return A * np.exp(-((x) ** 2) / (2 * sigma**2))

        # compute the Gaussian distribution value for each vertex
        gaussian_values = gaussian_func(geodesic_distances)
        gaussian_values = (gaussian_values - np.min(gaussian_values)) / (
            np.max(gaussian_values) - np.min(gaussian_values)
        )

        return gaussian_values / gaussian_values.sum()
