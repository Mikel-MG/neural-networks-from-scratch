from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, get_data_home


def load_mnist(use_cached: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load the MNIST digit dataset.

    It consists of 70000 images of handwritten digits (28x28 pixels, flattened), and their numeric labels.

    Args:
        use_cached (bool): Whether to use pre-cached content or not.

    Returns:
        tuple[np.ndarray, np.ndarray]: X and Y data.
    """

    # define the path where the cached arrays will be stored
    data_home_path = get_data_home()
    path_mnist_data = Path(data_home_path) / "mnist_data.npz"

    # if there is no cached dataset, or if it is not to be used
    if not path_mnist_data.exists() or not use_cached:
        # load mnist dataset (parsing is a bit slow)
        X_data, y_labels = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False,
            parser="liac-arff",
        )

        # reshape data
        y_labels = y_labels.astype(int)
        y_labels = y_labels.reshape(-1, 1)

        # save matrices in numpy format
        np.savez(
            path_mnist_data,
            X_data=X_data,
            y_labels=y_labels,
        )

    # if the cached dataset is found, load it (fast parsing)
    else:
        cached_data = np.load(path_mnist_data)
        X_data = cached_data["X_data"]
        y_labels = cached_data["y_labels"]

    return (X_data, y_labels)
