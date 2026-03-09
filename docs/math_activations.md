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
