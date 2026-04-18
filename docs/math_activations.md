# Activation functions

## Sigmoid

The sigmoid (logistic) function maps real-valued inputs to the interval $(0, 1)$, producing a smooth S-shaped curve. It is differentiable everywhere and its derivative can be written in terms of the function itself. The function saturates for large $|x|$, which can lead to very small gradients (vanishing gradient problem).

It is defined as

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Its derivative is

$$
\frac{d}{dx}\sigma(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)
$$

Here is the step-by-step derivation

$$
\begin{align*}
\sigma(x) &= \frac{1}{1 + e^{-x}} \\[1em]
\frac{d}{dx}\sigma(x) &= \frac{d}{dx}\left(\frac{1}{1 + e^{-x}}\right) = \frac{d}{dx}\left((1 + e^{-x})^{-1}\right) \\[1em]
&= (-1)(1 + e^{-x})^{-2}\cdot \frac{d}{dx}(1 + e^{-x}) = (-1)(1 + e^{-x})^{-2}\cdot(-e^{-x}) \\[1em]
&= \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} \\[1em]
&= \sigma(x)\frac{e^{-x}}{1+e^{-x}} = \sigma(x)\frac{1+e^{-x}-1}{1+e^{-x}} = \sigma(x)\left(\frac{1+e^{-x}}{1+e^{-x}} - \frac{1}{1+e^{-x}}\right) \\[1em]
&= \sigma(x)\bigl(1-\sigma(x)\bigr)
\end{align*}
$$

## Softmax

The softmax function maps a vector of real-valued inputs to a probability distribution over multiple classes. Each output lies in the interval (0,1), and all outputs sum to 1. It is differentiable everywhere, but unlike scalar activation functions (e.g. sigmoid), its derivatives are coupled across components.

It is defined as

$$
\text{softmax}(z_i) = S_i = \frac{e^{z_i}}{\sum_{k=1}^{K} e^{z_k}}
$$

where:

- $K$ is the number of outputs/classes
- $z_n$ is the *n*th component of the logit vector $\hat{z} = (z_1, z_2, z_3, \dots, z_K)$

The partial derivative of $S_i$ with respect to the $z_k$ is

$$
\frac{\partial S_i}{\partial z_k} = S_i (\delta_{ik} - S_k)
$$

Here is the step-by-step derivation

For convenience, we define

$$
Z = \sum_{k=1}^{K} e^{z_k}
$$

and thus

$$
\begin{align*}
S_i &= \frac{e^{z_i}}{Z} \\[1em]
\frac{\partial S_i}{\partial z_k} &= \frac{\partial}{\partial z_k}\left( \frac{e^{z_i}}{Z} \right) \\[1em]
&= \frac{\frac{\partial e^{z_i}}{\partial z_k} \cdot Z - e^{z_i} \cdot \frac{\partial Z}{\partial z_k}}{Z^2} \\[1em]
&= \frac{\delta_{ik}e^{z_i}\cdot Z - e^{z_i}\cdot e^{z_k}}{Z^2}
\end{align*}
$$

where $\delta_{ik}$ corresponds to the Kronecker delta notation

$$
\delta_{ik} =
\begin{cases}
1 & \text{if } i = k \\
0 & \text{if } i \ne k
\end{cases}
$$

Now, we can manipulate

$$
\begin{align*}
\frac{\partial S_i}{\partial z_k} &= \frac{\delta_{ik}e^{z_i}\cdot Z - e^{z_i}\cdot e^{z_k}}{Z^2} \\[1em]
&= \frac{e^{z_i}}{Z^2} (\delta_{ik}\cdot Z - e^{z_k}) \\[1em]
&= \frac{e^{z_i}}{Z} \cdot \frac{\delta_{ik}\cdot Z - e^{z_k}}{Z} \\[1em]
&= \frac{e^{z_i}}{Z} \cdot \left( \frac{\delta_{ik}\cdot Z}{Z} - \frac{e^{z_k}}{Z} \right) \\[1em]
&= S_i(\delta_{ik} - S_k)
\end{align*}
$$

Note that the derivative of $S$ with respect to the multi-component logit vector $\hat{z}$ is a $K \times K$ matrix, called the Jacobian of the softmax function.

$$
J =
\begin{bmatrix}
\frac{\partial S_1}{\partial z_1} & \frac{\partial S_1}{\partial z_2} & \cdots & \frac{\partial S_1}{\partial z_K} \\
\frac{\partial S_2}{\partial z_1} & \frac{\partial S_2}{\partial z_2} & \cdots & \frac{\partial S_2}{\partial z_K} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial S_K}{\partial z_1} & \frac{\partial S_K}{\partial z_2} & \cdots & \frac{\partial S_K}{\partial z_K}
\end{bmatrix}
$$

The Jacobian can be written in a compact vectorized form using only the output vector $S$.

$$
J = \mathrm{diag}(S) - S S^{\top}
$$

where $\mathrm{diag}(S)$ is the diagonal matrix with entries of S.