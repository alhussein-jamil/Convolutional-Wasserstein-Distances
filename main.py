from src.utils import load_normalized_image as load_image
import matplotlib.pyplot as plt
from src.wasserstein import wass_bary_2d
import numpy as np
import imageio
from pathlib import Path
import tempfile
from src.utils import load_BW_portrait, interpolate_bw_images
from src.mesh import Mesh, xyz_to_i
import trimesh
import pickle

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


def mesh_transform():
    meshes_output_dir = output_dir / "meshes" 
    meshes_output_dir.mkdir(exist_ok=True, parents=True)
    #number of cuts in each axe
    n = 40
    meshes = []
    meshes_names =[]
    meshes_path = "data/meshes"
    for file in Path(meshes_path).iterdir():
        name = file.name.split(".")[0]
        name = name.replace("-", "_").replace(" ", "_")
        meshes_names+=[name]
        exec(name+" = "+"Mesh(n=n)")
        meshes+=[eval(name)]
        exec(name+".mesh = trimesh.load(file)")

    scene = trimesh.Scene([meshes[meshes_names.index("duck")].mesh])
    scene.show()

    n_points_thousand = 6

    #initial number of points for monte-carlo
    n_pts = 1000

    for m in meshes:
        m.normalize_mesh()
        m.find_distribution(n,n_pts)

        
    for i,m in enumerate(meshes):
        print("Calculating distribution for "+meshes_names[i])
        if((meshes_output_dir / Path(meshes_names[i]+str(n_points_thousand)+'.pkl')).is_file()):
            print("already exists")
            with open(meshes_output_dir / (meshes_names[i]+str(n_points_thousand)+'.pkl'), 'rb') as f:
                meshes[i]=pickle.load(f)
        else:
            for _ in range(n_points_thousand):
                m.update_distribution(10000)
            for _ in range(3):
                m.fillBin()
            with open(meshes_output_dir / Path(meshes_names[i]+str(n_points_thousand)+'.pkl'), 'wb') as f:
                pickle.dump(meshes[i], f)

if __name__ == "__main__":
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # shape_transforms()
    # portrait_transform()
    mesh_transform()