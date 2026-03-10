import numpy as np


def generate_linear_data(
    m: float,
    b: float,
    n_samples: int,
    xmin: int = 0,
    xmax: int = 10,
    noise_scale: float = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates x and y coordinates that can be fitted to a linear model

    Args:
        m (float): Slope of the linear model that generates the data.
        b (float): Bias of the linear model that generates the data.
        n_samples (int): Number of data points to generate.
        xmin (int): Minimum X coordinate of the generated data.
        xmax (int): Maximum X coordinate of the generated data.
        noise_scale (float): Scale of the random noise.

    Returns:
        tuple[np.ndarray, np.ndarray]: X and Y data.
    """
    # generate data
    X = np.linspace(xmin, xmax, n_samples)
    noise = np.random.normal(scale=noise_scale, size=n_samples)
    Y = X * m + b + noise
    # reshape data
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    return (X, Y)
