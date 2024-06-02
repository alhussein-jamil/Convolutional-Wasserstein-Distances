import trimesh.exchange
import trimesh.exchange.export
from src.utils import load_normalized_image as load_image
import matplotlib.pyplot as plt
from src.wasserstein import wass_bary_2d, wass_bary_3d
import numpy as np
import imageio
from pathlib import Path
import tempfile
from src.utils import (
    load_BW_portrait,
    interpolate_bw_images,
    distribution_to_cloud,
    point_cloud_plot,
    plot_3d_binary,
    plot_cubic_mesh,
    distribution_to_binary,
    smoothed_figure_sub,
)
from src.mesh import Mesh
import trimesh
import pickle
import plotly.subplots as sp
import plotly.graph_objects as go


def shape_transforms():
    # Load the images
    path = "data/images/shapes/"
    dots = load_image(path + "shape2filled.png")
    star = load_image(path + "shape3filled.png")
    fivestar = load_image(path + "shape4filled.png")
    circle = load_image(path + "shape1filled.png")

    n = star.shape[0]
    dots = dots.reshape(-1)
    star = star.reshape(-1)
    circle = circle.reshape(-1)
    fivestar = fivestar.reshape(-1)
    grid_size = 5
    _, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(11, 11))

    for i in range(grid_size):
        for j in range(grid_size):
            coef = (
                1.0
                / (grid_size - 1) ** 2
                * np.array(
                    [
                        (grid_size - 1 - j) * (grid_size - 1 - i),
                        j * (grid_size - 1 - i),
                        j * i,
                        (grid_size - 1 - j) * i,
                    ]
                )
            )
            bary = wass_bary_2d(
                mus=[circle, star, fivestar, dots],
                coef=coef,
                a=np.ones((n**2)) / (0.0 + n**2),
                n=n,
                gamma=0.0015,
                iterations=3,
                entropic_sharpening=True,
            )
            bary = bary.reshape(n, n)
            threshold = 1e-6
            bary = np.where(bary > threshold, 1, 0)
            axes[i, j].imshow(bary, cmap="binary")
            axes[i, j].set_axis_off()  # turn off axes
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0
            )  # adjust spacing

    plt.savefig(output_dir / "shapes.png", bbox_inches="tight")
    plt.show()

    n_images = 20

    images = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(n_images):
            bary = wass_bary_2d(
                mus=[dots, star],
                coef=[1.0 / n_images * i, (1 - 1.0 / n_images * i)],
                a=np.ones((n**2)) / (0.0 + n**2),
                n=n,
                gamma=0.0015,
                iterations=3,
                entropic_sharpening=False,
            )
            bary = bary.reshape(n, n)
            threshold = (1e-6) * 5
            bary = np.where(bary > threshold, 1, 0)
            # Plot the heatmap using imshow
            plt.imshow(bary, cmap="binary")
            plt.axis("off")  # turn off axes to make gif cleaner
            plt.savefig(
                Path(temp_dir) / f"image_{i}.png", bbox_inches="tight"
            )  # save figure as png
            plt.close()
            images.append(
                imageio.imread(Path(temp_dir) / f"image_{i}.png")
            )  # add image to list for gif

        gif_path = "dots_to_star.gif"
        # create gif from list of images
        imageio.mimsave(output_dir / gif_path, images, duration=0.1)


def portrait_transform():
    epsilon = 0
    monge = load_BW_portrait("data/monge.png").reshape((1, -1)) + epsilon
    m = np.max(monge)  # np.ones((1,163**2))-
    monge = m - monge
    kant = load_BW_portrait("data/kantorowich.png").reshape((1, -1)) + epsilon
    m = np.max(kant)
    kant = m - kant
    with tempfile.TemporaryDirectory() as temp_dir:
        images = interpolate_bw_images(
            mus=[monge, kant],
            n_interpolations=20,
            gamma=0.0002,
            n_iter=100,
            temp_dir=temp_dir,
        )
        imageio.mimsave("output/portrait_monge_kantor.gif", images, duration=0.1)


def init_meshes(n=40) -> tuple[list[Mesh], list[str]]:
    meshes_output_dir = output_dir / "meshes"
    meshes_output_dir.mkdir(exist_ok=True, parents=True)
    # number of cuts in each axe
    meshes: list[Mesh] = []
    meshes_names = []
    meshes_path = "data/meshes"
    for file in Path(meshes_path).iterdir():
        name = file.name.split(".")[0]
        name = name.replace("-", "_").replace(" ", "_")
        meshes_names += [name]
        exec(name + " = " + "Mesh(n=n)")
        meshes += [eval(name)]
        exec(name + ".mesh = trimesh.load(file)")

    scene = trimesh.Scene([meshes[meshes_names.index("duck")].mesh])
    scene.show()

    n_points_thousand = 6

    # initial number of points for monte-carlo
    n_pts = 1000

    for m in meshes:
        m.normalize_mesh()
        m.find_distribution(n, n_pts)

    for i, m in enumerate(meshes):
        print("Calculating distribution for " + meshes_names[i])
        if (
            meshes_output_dir / Path(meshes_names[i] + str(n_points_thousand) + ".pkl")
        ).is_file():
            print("already exists")
            with open(
                meshes_output_dir / (meshes_names[i] + str(n_points_thousand) + ".pkl"),
                "rb",
            ) as f:
                meshes[i] = pickle.load(f)
        else:
            for _ in range(n_points_thousand):
                m.update_distribution(10000)
            for _ in range(3):
                m.fillBin()
            with open(
                meshes_output_dir
                / Path(meshes_names[i] + str(n_points_thousand) + ".pkl"),
                "wb",
            ) as f:
                pickle.dump(meshes[i], f)

    return meshes, meshes_names


def mesh_transform(meshes: list[Mesh], meshes_names: list[str], n: int = 40) -> None:
    point_cloud_plot(
        distribution_to_cloud(
            meshes[meshes_names.index("duck")].distribution.reshape(n, n, n), scale=3
        )
    )
    plot_3d_binary(meshes[meshes_names.index("duck")].binary_distribution, n)

    plot_cubic_mesh(meshes[meshes_names.index("duck")].binary, n)

    a = [1.0 / (n * n * n)] * (n * n * n)
    gamma = 0.0015
    bin = []

    # Define the number of subplots
    grid_size = 4
    num_plots = grid_size**2

    choosen_meshes = [np.random.randint(len(meshes)) for _ in range(4)]
    choosen_meshes_names = ["duck", "torus", "mushroom", "sphere_102"]
    # choosen_meshes_names = random.sample(meshes_names,4)
    foldername = "".join(choosen_meshes_names) + str(grid_size)
    gen_meshses_output_dir = Path("output/generated/" + foldername)
    gen_meshses_output_dir.mkdir(exist_ok=True, parents=True)

    choosen_meshes = [meshes_names.index(name) for name in choosen_meshes_names]

    coefs = []
    for i in range(grid_size):
        for j in range(grid_size):
            coef = (
                1.0
                / (grid_size - 1) ** 2
                * np.array(
                    [
                        (grid_size - 1 - j) * (grid_size - 1 - i),
                        j * (grid_size - 1 - i),
                        j * i,
                        (grid_size - 1 - j) * i,
                    ]
                )
            )
            coefs += [coef]
            bin += [
                wass_bary_3d(
                    mus=np.array(
                        [meshes[k].binary_distribution for k in choosen_meshes]
                    ),
                    coef=coef,
                    n=n,
                    a=a,
                    iterations=3,
                    prefactorized=False,
                    conv=True,
                    gamma=gamma,
                )
            ]
    coefs = np.array(coefs).round(1)
    # plot_cubic_mesh(fillBin(distribution_to_bin(x)))
    # smoothed_fig(fillBin(distribution_to_bin(x)))
    # Create a new figure with subplots
    fig = sp.make_subplots(
        rows=grid_size,
        cols=grid_size,
        specs=[[{"type": "scene"}] * grid_size] * grid_size,
    )

    # Loop over the subplots
    for i in range(num_plots):
        # Get the subplot ID
        plot_id = (i // grid_size + 1, i % grid_size + 1)
        # Create the smoothed figure subplot
        subplot, mesh = smoothed_figure_sub(
            distribution_to_binary(bin[i]), plot_id=plot_id, n=n
        )
        (gen_meshses_output_dir / foldername).mkdir(exist_ok=True, parents=True)
        filename = (
            str(coefs[i][0])
            + choosen_meshes_names[0]
            + "_"
            + str(coefs[i][1])
            + choosen_meshes_names[1]
            + "_"
            + str(coefs[i][2])
            + choosen_meshes_names[2]
            + "_"
            + str(coefs[i][3])
            + choosen_meshes_names[3]
        )
        trimesh.exchange.export.export_mesh(
            mesh,
            gen_meshses_output_dir / foldername / (filename + ".off"),
            file_type="off",
        )
        # Add the subplot to the figure
        fig.add_trace(subplot.data[0], row=plot_id[0], col=plot_id[1])

    # Update the layout of the subplots
    fig.update_layout(height=1600, width=1600, title_text=foldername)
    fig.show()


def gaussian_transform(
    meshes: list[Mesh],
    meshes_names: list[str],
    n: int = 40,
    choosen_mesh_name: str = "duck",
) -> None:
    used_mesh = meshes[meshes_names.index(choosen_mesh_name)]
    used_mesh.cotangent_laplacian_with_area()
    gaussain1 = used_mesh.gaussain_on_mesh(0)
    gaussain2 = used_mesh.gaussain_on_mesh(150)
    mesh: trimesh.Trimesh = used_mesh.mesh
    # create a Plotly 3D surface plot of the sphere with colors based on the Gaussian distribution values
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                intensity=gaussain1 + gaussain2,
                colorscale="Viridis",
            )
        ]
    )
    fig.show()


if __name__ == "__main__":
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # shape_transforms()
    # portrait_transform()
    n = 40
    meshes, meshes_names = init_meshes(n)
    # mesh_transform(meshes, meshes_names, n)
    gaussian_transform(meshes, meshes_names, n, "duck")
