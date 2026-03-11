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


def generate_two_moons(
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates x and y coordinates that represent two partially complementary half-moons.

    Args:
        n_samples (int): Number of data points to generate.

    Returns:
        tuple[np.ndarray, np.ndarray]: X and Y data.
    """
    theta = np.random.rand(n_samples) * np.pi

    #  generate data for the first moon
    x1 = np.cos(theta)
    y1 = np.sin(theta)

    #  generate data for the second moon
    x2 = 1 - x1
    y2 = -y1 + 0.5

    # stack data
    X = np.vstack([np.stack([x1, y1], axis=1), np.stack([x2, y2], axis=1)])
    Y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # add noise
    noise = np.random.rand(X.shape[0], 2) / 1.5
    X += noise

    # scale vertical dimension
    X[:, 1] *= 2

    # reshape data
    Y = Y.reshape(-1, 1)

    return (X, Y)
