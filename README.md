# Julia Framework for Interior-Point Trust-Region Method for Composite Optimization
This package contains code to solve composite optimization problems of the form

minₓf(x) + h(x)
s.t. c(x)≦0

where f(x) is smooth and h(x) is nonsmooth and possibly nonconvex.

## Installation
To install the package, hit `]` from the Julia command line to enter the package manager and type
```julia
pkg> add https://github.com/UW-AMO/TRNC
```

## Usage & Options
With `using TRNC` the package exports `barrier_alg` which runs the both an simple log-barrier method for interior point problems and the interior point problem itself. The user has to provide a method to solve the inner descent direction, but several of these methods are available with the package, namely minConf_SPG, proximal gradient, FISTA, and a variable splitting method.

### Simple Least-Squares Example
Algorithm 4.2 (Non-smooth Trust Region) is implemented as `IntPt_TR` in the file `IP_alg.jl`, and Algorithm 4.3 is implemented as `barrier_alg` in the file `barrier.jl`. To run an easy example, simply type
```julia
julia LS_ex.jl
```
to solve the problem minₓ ||Ax - b||² s.t. 0≦x≦1 for randomly generated 200x100 matrix A and gaussian noise observations b and 0≦x≦1. This script allows the user to define the function to solve in the same script and pass said function to `IntPt_TR`, which takes in that function, some initial parameter guesses for x, zl, zu, and a bank of options for `IntPt_TR`. Make sure you have Julia 1.0+.

### Basis Pursuit Example
To demonstrate the nonsmooth capabilities of our algorithm, we can also solve the classic Tichonoff version of the basis pursuit de-noise example (or LASSO) minₓ ||Ax - b||² +λ||x||₁ -10≦x≦10. Similarly to the LS example, we randomly generated 200x100 matrix A and gaussian noise observations b. We solve for the descent direction with our variable splitting method. To run this example for yourself, type
```julia
julia bpdn_ex.jl
```

### Algorithm 4.3: Barrier Algorithm
As currently implemented, this algorithm is pretty straight forward. It simply uses a log-barrier method with user-defined initial parameter to enforce the conditions on the minimization problem. Currently, μ is shrunken by a constant factor of 1/2 every time the inner interior point algorithm finds its feasible solution for the previous μ. There is a default stopping tolerance of μ>1e-10$, but the user can define a different μ lower bound if desired. To do: input convex $\mu$ as duality gap?



### Algorithm 4.2: Non-smooth Trust Region
In the draft, Algorithm 4.2 is represented by `IntPt_TR`, a function that is on the julia file `IP_alg.jl`. While more about the function is described in the aforementioned file, it basically takes some initial parameter guesses as well as options that are a mutable struct defined on that same julia file (i.e. functions to find the descent direction, prox of non-smooth descent direction function, etc). See `IP_alg.jl` for more details. There are two main functions that this algorithm is currently dependent upon: `Qcustom.jl` and `DescentMethods.jl`. This julia file has FISTA, prox-gradient, and variable splitting implementations that one can use to solve for the descent direction in the interior point algorithm. For the simple examples where the non-smooth component h(x) is zero, one can use `minconf_spg`. `Minconf_spg` is a spectral minimization algorithm written by Michael Freidlander & company that is quickly able to project onto the trust-region norm ball one decides to use when solving the descent direction (listed as s in both the paper draft and our code). For some norms, one can define the lp-norm ball projection; for the l1-norm ball, projection is done with the `OneProjector.jl` file, which is currently only for scalar-valued l1 norms (also courtesy of Michael Freidlander & company). The other file, `Qcustom.jl`, is simply the 2nd order approximation of the objective function; the user supplies the gradient and hessian in another "params" struct. As of right now, these are produced whenever the first objective function is called.


Complete list of functions and option structures:
* `barrier_alg`
  * options go directly into the function argument
* `IntPt_TR`
  * `IP_struct`- takes in user-defined f and h as mandatory arguments, and optional problem arguments dealing primarily with algorithmic choices. These include:
    * `l` - lower bound (defaults to -∞)
    * `u` - upper bound (defaults to +∞)
    * `s_alg` - algorithm user-supplied method to solve for the descent direction (defaults to Variable Projection INSERT CITATION)
    * `FO_options` - algorithm options for user-supplied method to solve for the descent direction
    * `χ_projector` - projection onto trust-region norm ball (defaults to l₁-norm ball)
    * `ψk` - model for nonsmooth function h (defaults to h; the distinction between the two is for calculating ρ)
    * `prox_ψk` - proximal operator for h (defaults to h)
  * `IP_options` - numerical options for the interior point algorithm
    * `epsC` - KKT termination criteria for barrier criteria (CITE PAPER RELATIONSHIP)
    * `epsD` - KKT termination criteria for gradient criteria (CITE PAPER RELATIONSHIP)
    * `ptf` - print frequency
    * `simple` - switch for if h=0, then we have an easy projection method
    * `maxIter` - maximum number of Interior Point iterations (note: does not include barrier iterations, resets to zero after each barrier parameter μ update)
    * `Δk` - initial trust region radius (defaults to 1)
* `DescentMethods.jl` - various first order methods for solving the descent direction problem
  * `FISTA, FISTA!` - implementation of FISTA
  * `PG, PG!` - standard proximal gradient
  * `prox_split_2w` - variable splitting method for approximate proximal function solves (CITE BPDN PAPER)
  * `s_options` - various options and tolerances for the above first order methods. All have the same structure, with the variable projection having a few extra options.
    * `β` - only required initialization option, surrogate for step length ν^-1 in paper. Usually set to some approximation of the Lipschitz constant.
    * `optTol` - termination tolerance for ||x⁺ - x|| (defaults to 1e-10)
    * `maxIter` - # of iterations (defaults to 10000)
    * `verbose` - level of print frequency: 0 = never, 1 = 10 total, 2 = 100 total (default), 3+ = every iteration
    * -Variable Splitting Only-
      * `η` - split parameter weight (default is 1); this effectively closes the distance between the projected variables and the dummy variables
      * `η_factor` - factor to decrease η by; ie η = .9η (default)
      * `restart` - # of times to apply η_factor; note this occurs when the algorithm exits for a particular η
      * `σ_TR` - TR ball radius (defaults to 1.0)
      * `WoptTol` - ||w⁺ - w|| tolerance (defaults to 1e-10)
      * `gk` - gradient vector
      * `Bk` - Hessian/Hessian approximation
      * `xk` - current x in interior point method


## References
1. D. Orban, A. Aravkin, and R. Baraldi (2020), [*An Interior Point Trust Region Method for Composite Optimization*] (in development)
2. R. Baraldi, R. Kumar, and A. Aravkin (2019), [*Basis Pursuit De-noise with Non-smooth Constraints*](10.1109/TSP.2019.2946029), IEEE Transactions on Signal Processing, vol. 67, no. 22, pp. 5811-5823.

<!-- In summary, the code structure is:
				Calls       |      Defines
				--------------------------
- barrier.jl: IntPt_TR		
- IPscript.jl:  IntPt_TR	  objective
				IP_params

- IP_alg.jl:    minconf_spg   IntPt_TR
				OneProjector  IP_params
				Qcustom

- minconf_spg/: OneProjector  minconf_spg
				barrier obj   spg_params

- Qcustom.jl:   grad, Hess    QCustom
							  Q_params -->
