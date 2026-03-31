# Loss functions

## MSE

The mean squared error (MSE) loss function measures the average squared difference between the ground truth values ($y$) and the predicted values ($\hat{y}$). It is differentiable everywhere and yields a gradient that is linear in the prediction error.

It is defined as

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

Its derivative with respect to the prediction for sample $i$ is

$$
\frac{\partial \text{MSE} }{\partial \hat{y}_i } = \frac{2}{n} (\hat{y}_i - y_i)
$$
