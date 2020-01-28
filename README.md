# Julia Framework for Interior-Point Trust-Region Method for Composite Optimization
This package contains code to solve problems of the form
$
\min_x \ f(x) + h(x)\\
\text{s.t.} c(x)\leq 0
$
where $f(x)$ is smooth and $h(x)$ is nonsmooth and possibly nonconvex.

## Installation
To install the package, hit `]` from the Julia command line to enter the package manager and type
```julia
pkg> add https://github.com/UW-AMO/TRNC
```

## Usage
With `using TRNC` the package exports `barrier_alg` which runs the both the barrier method for interior point problems and the interior point problem itself. One also gets several option structures:
* `barrier_alg`
  * options go directly into the function argument
* `IntPt_TR`
  * `IP_struct`- takes in user-defined $f$ and $h$ as mandatory arguments, and optional problem arguments dealing primarily with algorithmic choices. These include:
    * `l` - lower bound (defaults to $-\infty$)
    * `u` - upper bound (defaults to $+\infty$)
    * `s_alg` - algorithm user-supplied method to solve for the descent direction (defaults to Variable Projection INSERT CITATION)
    * `FO_options` - algorithm options for user-supplied method to solve for the descent direction
    * `χ_projector` - projection onto trust-region norm ball (defaults to $\ell_1$-norm ball)
    * `ψk` - model for nonsmooth function $h$ (defaults to $h$; the distinction between the two is for calculating $\rho$)
    * `prox_ψk` - proximal operator for $h$ (defaults to $h$)
  * `IP_options` - numerical options for the interior point algorithm
    * `epsC` - KKT termination criteria for barrier criteria (CITE PAPER RELATIONSHIP)
    * `epsD` - KKT termination criteria for gradient criteria (CITE PAPER RELATIONSHIP)
    * `ptf` - print frequency
    * `simple` - switch for if $h=0$, then we have an easy projection method
    * `maxIter` - maximum number of Interior Point iterations (note: does not include barrier iterations, resets to zero after each barrier parameter $\mu$ update)
    * `Δk` - initial trust region radius (defaults to 1)
* `DescentMethods` - various first order methods for solving the descent direction problem
  * `FISTA, FISTA!` - implementation of FISTA
  * `PG, PG!` - standard proximal gradient
  * `prox_split_2w` - variable splitting method for solving proximal functions (CITE BPDN PAPER)
  * `s_options` - various options and tolerances for the above first order methods. All have the same structure
    * `β` - only required initialization option, surrogate for step length $\nu$ in paper. Usually set to some approximation of the Lipschitz constant.
    * `optTol` - termination tolerance for $\|x^{k+1} - x^k\|$ (defaults to 1e-10)
    * `maxIter` - # of iterations (defaults to 10000)
    * `verbose` -
    * `η` -
    <!-- β;optTol=1f-10, maxIter=10000, verbose=2, restart=10, λ=1.0, η =1.0, η_factor=.9,σ_TR=1.0, -->
         <!-- WoptTol=1f-10, gk = Vector{Float64}(undef,0), Bk = Array{Float64}(undef, 0,0), xk=Vector{Float64}(undef,0) -->
## Overview
This folder contains code to run IP TR methods for non-smooth (potentially) non-convex composite optimization problems. Algorithm 4.2 (Non-smooth Trust Region) is implemented as IntPt_TR in the file "IP_alg.jl", and Algorithm 4.3 is implemented as barrier_alg in the file "barrier.jl". To run an easy example, simply type
```julia
julia LS_ex.jl
```
to solve the problem $\min_x \ \|Ax - b\|^2$ for randomly generated 200x100 matrix $A$ and gaussian noise observations $b$ and $0\leq x \leq 1$. This script allows the user to define the function to solve in the same script and pass said function to IntPt_TR, which takes in that function, some initial parameter guesses for $x, z_l, z_u$, and a bank of options for IntPt_TR. Make sure you have Julia 1.0+.


## Algorithm 4.3: Barrier Algorithm
As currently implemented, this algorithm is pretty straight forward. It simply uses a log-barrier method with user-defined initial parameter to enforce the conditions on the minimization problem. Currently, $\mu$ is shrunken by a constant factor of 1/2 every time the inner interior point algorithm finds its feasible solution for the previous $\mu$. There is a default tolerance of $\mu>1e-10$, but the user can define a different $\mu$ lower bound if desired. To do: input convex $\mu$ as duality gap?



## Algorithm 4.2: Non-smooth Trust Region
In the draft, algorithm 4.2 is represented by IntPt_TR, a function that is on the julia file IP_alg.jl. While more about the function is described in the aforementioned file, it basically takes some initial parameter guesses as well as options that are a mutable struct defined on that same julia file (i.e. functions to find the descent direction, prox of non-smooth descent direction function, etc). See IP_alg.jl for more details. There are two main functions that this algorithm is currently dependent upon: Qcustom.jl and DescentMethods.jl. This julia file has FISTA, prox-gradient, and variable projection implementations that one can use to solve for the descent direction in the interior point algorithm. For the simple examples where the non-smooth component $R_k(s)$ is zero, one can use minconf_spg. Minconf_spg is a spectral minimization algorithm written by Michael Freidlander and company that is quickly able to project onto the trust-region norm ball one decides to use when solving the descent direction (listed as $s$ in both the paper draft and our code). For some norms, one can define the $l_q$-norm ball projection; for the $l_1$-norm ball, projection is done with the OneProjector.jl file, which is currently only for scalar-valued l1 norms (also courtesy of Michael Freidlander and company). The other file, Qcustom.jl, is simply the 2nd order approximation of the objective function; the user supplies the gradient and hessian in another "params" struct. As of right now, these are produced whenever the first objective function is called.

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
