# [Algorithms](@id algorithms)

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

The following table gives an overview of the available algorithms in the general case.

Algorithm | Quadratic Regularization | Trust Region | Quadratic term for $\varphi$ : H | Reference
----------|--------------------------|--------------|---------------|----------
[`R2`](@ref R2) | Yes | No | $H = 0$  | [aravkin-baraldi-orban-2022; Algorithm 6.1](@cite)
[`R2N`](@ref R2N) | Yes | No | Any Symmetric| [diouane-habiboullah-orban-2024; Algorithm 1](@cite)
[`R2DH`](@ref R2DH) | Yes | No | Any Diagonal | [diouane-habiboullah-orban-2024; Algorithm 1](@cite)
[`TR`](@ref TR) | No | Yes | Any Symmetric | [aravkin-baraldi-orban-2022; Algorithm 3.1](@cite)
[`TRDH`](@ref TRDH) | No | Yes | Any Diagonal | [leconte-orban-2025; Algorithm 5.1](@cite)

## Nonlinear least-squares
This package provides two algorithms, [`LM`](@ref LM) and [`LMTR`](@ref LMTR), specialized for regularized, nonlinear least-squares.
That is, problems of the form
```math
\underset{x \in \mathbb{R}^n}{\text{minimize}} \quad \frac{1}{2}\|F(x)\|_2^2 + h(x),
```
where $F : \mathbb{R}^n \mapsto \mathbb{R}^m$ is continuously differentiable and $h : \mathbb{R}^n \mapsto \mathbb{R} \cup \{\infty\}$ is lower semi-continuous.
In that case, the model $\varphi$ is defined as 
```math
\varphi(s; x) = \frac{1}{2}\|F(x) + J(x)s\|_2^2,
```
where $J(x)$ is the Jacobian of $F$ at $x$.
Similar to the algorithms in the previous section, we either add a quadratic regularization to the model ([`LM`](@ref LM)) or a trust-region ([`LMTR`](@ref LMTR)).
These algorithms are described in [aravkin-baraldi-orban-2024](@cite).

## Constrained Optimization
