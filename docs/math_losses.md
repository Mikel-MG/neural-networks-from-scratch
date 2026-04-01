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

## BCE

The binary cross entropy (BCE) loss function measures the average discrepancy between the ground truth labels ($y$) and the predicted probabilities ($\hat{y}$) for binary classification. It is differentiable for $\hat{y} \in (0,1)$ and yields a gradient that increases sharply as predictions become confidently incorrect.

It is defined as 

$$
\text{BCE} = - \frac{1}{n} \sum_{i=1}^{n} \big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \big]
$$

Its derivative with respect to the prediction for sample $i$ is

$$
\frac{\partial \text{BCE} }{\partial \hat{y}_i } = - \frac{y_i}{\hat{y_i}}  + \frac{1-y_i}{1-\hat{y}_i}
$$

Here is the step-by-step derivation

$$
\begin{align*}
\text{BCE} &= - \big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \big]\\[1em]
\frac{\partial \text{BCE} }{\partial \hat{y}_i} &= - \left[ y_i \cdot \frac{\partial \log(\hat{y}_i)}{\partial \hat{y}_i}
+ (1 - y_i) \cdot \frac{\partial \log(1 - \hat{y}_i)}{\partial \hat{y}_i} \right]\\[1em]
&= - \left[ y_i \cdot \frac{1}{\hat{y}_i} + (1-y_i) \cdot \frac{-1}{(1 - \hat{y}_i)} \right]\\[1em]
&= - \frac{y_i}{\hat{y}_i} + \frac{1-y_i}{1-\hat{y}_i}
\end{align*}
$$

???+ note
    In practice, this direct per-sample derivative is rarely applied on its own. It is typically combined with a **sigmoid activation** function applied to the model output, producing a single, simplified gradient:
    
    $$\frac{\partial \text{BCE}}{\partial z_i} = \hat{y}_i - y_i$$
    
    This combined form improves both **computational efficiency** and **numerical stability** during training.