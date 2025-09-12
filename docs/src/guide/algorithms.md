# Algorithms

## General case
The algorithms in this package are based upon the approach of [aravkin-baraldi-orban-2022](@cite).
Suppose we are given the general regularized problem
```math
\underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x) + h(x),
```
where $f : \mathbb{R}^n \mapsto \mathbb{R}$ is continuously differentiable and $h : \mathbb{R}^n \mapsto \mathbb{R} \cup \{\infty\}$ is lower semi-continuous.
Instead of solving the above directly, which is often impossible, we will solve a simplified version of it repeatedly until we reach a minimizer of the problem above.
To do so, suppose we are given an iterate $x_0 \in \mathbb{R}^n$, we wish to compute a step, $s_0 \in \mathbb{R}^n$ and improve our iterate with $x_1 := x_0 + s_0$.
Now, we are going to approximate the functions $f$ and $h$ around $x_0$ with simpler functions (models), which we denote respectively $\varphi(\cdot; x_0)$ and $\psi(\cdot; x_0)$ so that
```math
\varphi(s; x_0) \approx f(x_0 + s) \quad \text{and} \quad \psi(s; x_0) \approx h(x_0 + s). 
```
We then wish to compute the step as
```math
s_0 \in \underset{s \in \mathbb{R}^n}{\argmin} \  \varphi(s; x_0) + \psi(s; x_0).
```
In order to ensure convergence and to handle the potential nonconvexity of the objective function, we either add a trust-region,
```math
s_0 \in \underset{s \in \mathbb{R}^n}{\argmin} \  \varphi(s; x_0) + \psi(s; x_0) \quad \text{subject to} \ \|s\| \leq \Delta,
```
or a quadratic regularization
```math
s_0 \in \underset{s \in \mathbb{R}^n}{\argmin} \  \varphi(s; x_0) + \psi(s; x_0) + \sigma \|s\|^2_2.
```
Algorithms that work with a trust-region are [`TR`](@ref TR) and [`TRDH`](@ref TRDH) and the ones working with a quadratic regularization are [`R2`](@ref R2), [`R2N`](@ref R2N) and [`R2DH`](@ref R2DH)

The models for the smooth part `f` in this package are always quadratic models of the form
```math
\varphi(s; x_0) = f(x_0) + \nabla f(x_0)^T s + \frac{1}{2} s^T H(x_0) s,
```
where $H(x_0)$ is a symmetric matrix that can be either $0$, the Hessian of $f$ (if it exists) or a quasi-Newton approximation.
Some algorithms require a specific structure for $H$, for an overview, refer to the table below.

For the model of the regularizer $h$, it is less straightforward.
First, we require the model to locally approximate $h$ sufficiently well, see [aravkin-baraldi-orban-2022; Model Assumption 3.2.](@cite)
Then, the model $\psi$ should be such that the shifted proximal mapping
```math
\text{prox}_{\sigma^{-1}\psi} (q) := \underset{s \in \mathbb{R}^n}{\argmin} \psi(s;x_0) + \frac{\sigma}{2} \| s - q \|_2^2
```
can be computed efficiently. 

!!! note
    While the user can choose whichever regularizer he wants and then choose an appropriate model for it, he also needs to implement the function `prox!(y, ψ, q, σ)` which computes the proximal mapping in place. 

For example, if the regularizer is the Euclidean norm, $h = \|cdot\|_2$, then we can simply choose the model as the function itself, $\psi(s ;x) := h(x + s)$ because
```math
\underset{s \in \mathbb{R}^n}{\argmin} \ \|x + s\|_2 + \frac{\sigma}{2} \| s - q \|_2^2 = (1 - \frac{1}{\sigma\|x + q\|})_+ (x + q) - x.
```
where $(\cdot)_+ := \max \{0, \cdot \}$.

Another example is when the regularizer is the Euclidean norm composed with a function, $h = \|c(\cdot)\|_2$.
In that case, taking the model as the function itself would make the proximal mapping intractable. 
Instead, one can choose the model $\psi(s ;x) := \|c(x) + J(x)s\|_2$, which is a composition of an affine function with the Euclidean norm. 
This approach was used in [diouane-gollier-orban-2024](@cite).

For more information on custom regularizers and models thereof, please refer to [this section](@ref custom_regularizers).

The following table gives an overview of the available algorithms in the general case.

Algorithm | Quadratic Regularization | Trust Region | Quadratic term for $\varphi$ : H | Reference
----------|--------------------------|--------------|---------------|----------
[`R2`](@ref R2) | Yes | No | $H = 0$  | [aravkin-baraldi-orban-2022; Algorithm 6.1](@cite)
[`R2N`](@ref R2N) | Yes | No | Any Symmetric| [diouane-habiboullah-orban-2024; Algorithm 1](@cite)
[`R2DH`](@ref R2DH) | Yes | No | Any Diagonal | [diouane-habiboullah-orban-2024; Algorithm 1](@cite)
[`TR`](@ref TR) | No | Yes | Any Symmetric | [aravkin-baraldi-orban-2022; Algorithm 3.1](@cite)
[`TRDH`](@ref TRDH) | No | Yes | Any Diagonal | [leconte-orban-2025; Algorithm 5.1](@cite)

## Nonlinear least-squares

## Constrained Optimization
