import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Load the MNIST digit dataset.

    It consists of 70000 images of handwritten digits (28x28 pixels, flattened), and their numeric labels.

    Returns:
        tuple[np.ndarray, np.ndarray]: X and Y data.
    """

    # load mnist dataset
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

    return (X_data, y_labels)
