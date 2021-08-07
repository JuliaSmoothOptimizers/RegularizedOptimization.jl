# A Proximal Quasi-Newton Trust-Region Method for Regularized Optimization
This package contains code to solve regularized optimization problems of the form

minₓf(x) + h(x)

where f(x) is smooth and h(x) is nonsmooth; each can be nonconvex.

## Installation
To install the package, hit `]` from the Julia command line to enter the package manager and type
```julia
pkg> add https://github.com/UW-AMO/TRNC
```

## Usage & Options
TODO

## References
1. D. Orban, A. Aravkin, and R. Baraldi (2020), [*An Interior Point Trust Region Method for Composite Optimization*] (in development) TO COMPLETE
2. R. Baraldi, R. Kumar, and A. Aravkin (2019), [*Basis Pursuit De-noise with Non-smooth Constraints*](10.1109/TSP.2019.2946029), IEEE Transactions on Signal Processing, vol. 67, no. 22, pp. 5811-5823.