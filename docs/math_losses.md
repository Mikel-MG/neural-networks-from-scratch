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

## CCE

The categorical cross entropy (CCE) loss function measures the average discrepancy between the multi-class ground truth labels ($y$), which are typically one-hot encoded, and the predicted probability distribution ($\hat{y}$). It is differentiable for each $\hat{y}_{i,k} \in (0,1)$ and yields a gradient that increases sharply as predictions become confidently incorrect.

It is defined as

$$
\text{CCE} = - \frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})
$$

where:

- $K$ is the number of classes
- $y_{i,k}$ is the ground truth (1 if sample $i$ belongs to class $k$, else 0)
- $\hat{y}_{i,k}$ is the predicted probability for class $k$

Its derivative with respect to the prediction for sample $i$ and class $k$ is

$$
\frac{\partial \text{CCE}}{\partial \hat{y}_{i,k}} = - \frac{y_{i,k}}{\hat{y}_{i,k}}
$$

Here is the step-by-step derivation

$$
\begin{align*}
\text{CCE}_i &= - \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k}) \\[1em]
\frac{\partial \text{CCE}}{\partial \hat{y}_{i,k}}
&= - y_{i,k} \cdot \frac{\partial \log(\hat{y}_{i,k})}{\partial \hat{y}_{i,k}} \\[1em]
&= - \frac{y_{i,k}}{\hat{y}_{i,k}}
\end{align*}
$$

Note that the derivative with respect to the multi-class prediction vector $\hat{y}_i$ is a $K$-dimensional vector, not a scalar. For sample $i$

$$
\begin{align*}
\text{CCE}_i &= - \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k}) \\[1em]
\frac{\partial \text{CCE}_i}{\partial \hat{y}_i}
&= - \begin{bmatrix}
\frac{\partial}{\partial \hat{y}_{i,1}} \left( y_{i,1} \log(\hat{y}_{i,1}) \right) \\
\frac{\partial}{\partial \hat{y}_{i,2}} \left( y_{i,2} \log(\hat{y}_{i,2}) \right) \\
\vdots \\
\frac{\partial}{\partial \hat{y}_{i,K}} \left( y_{i,K} \log(\hat{y}_{i,K}) \right)
\end{bmatrix} \\[1em]
&= - \begin{bmatrix}
y_{i,1} \cdot \frac{1}{\hat{y}_{i,1}} \\
y_{i,2} \cdot \frac{1}{\hat{y}_{i,2}} \\
\vdots \\
y_{i,K} \cdot \frac{1}{\hat{y}_{i,K}}
\end{bmatrix} \\[1em]
&= - \begin{bmatrix}
\frac{y_{i,1}}{\hat{y}_{i,1}} \\
\frac{y_{i,2}}{\hat{y}_{i,2}} \\
\vdots \\
\frac{y_{i,K}}{\hat{y}_{i,K}}
\end{bmatrix}
\end{align*}
$$

Equivalently, in compact vector notation:

$$
\frac{\partial \text{CCE}_i}{\partial \hat{y}_i}
= - \frac{y_i}{\hat{y}_i}
$$

where the division is element-wise.

???+ note

    In practice, this direct per-sample derivative is rarely applied on its own. It is typically combined with a **softmax activation** function applied to the model output, producing a single, simplified gradient:

    $$
    \frac{\partial \text{CCE}}{\partial z_{i,k}} = \hat{y}_{i,k} - y_{i,k}
    $$

    Analogously to the **BCE + sigmoid** case, this compact form improves both **computational efficiency** and **numerical stability** during training.
