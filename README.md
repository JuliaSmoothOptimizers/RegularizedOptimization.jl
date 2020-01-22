# Julia Framework for Interior-Point Trust-Region Method for Composite Optimization


## Overview
This folder contains code to run IP TR methods for non-smooth (potentially) non-convex composite optimization problems. Algorithm 4.2 (Non-smooth Trust Region) is implemented as IntPt_TR in the file "IP_alg.jl", and Algorithm 4.3 is implemented as barrier_alg in the file "barrier.jl". To run an easy example, simply type
```julia
julia LS_ex.jl
```
to solve the problem $$\min_x \ \|Ax - b\|^2$$ for randomly generated 200x100 matrix $$A$$ and gaussian noise observations $$b$$ and $$0\leq x \leq 1$$. This script allows the user to define the function to solve in the same script and pass said function to IntPt_TR, which takes in that function, some initial parameter guesses for $$x, z_l, z_u$$, and a bank of options for IntPt_TR. Make sure you have Julia 1.0+.


## Algorithm 4.3: Barrier Algorithm
As currently implemented, this algorithm is pretty straight-forward. It simply uses a log-barrier method with user-defined initial parameter to enforce the conditions on the minimization problem. Currently, $\mu$ is shrunken by a constant factor of 1/2 every time the inner interior point algorithm finds its feasible solution for the previous $\mu$. There is a default tolerance of $\mu>1e-10$, but the user can define a different $\mu$ lower bound if desired. To do: input convex $\mu$ as duality gap?



## Algorithm 4.2: Non-smooth Trust Region
In the draft, algorithm 4.2 is represented by IntPt_TR, a function that is on the julia file IP_alg.jl. While more about the function is described in the aforementioned file, it basically takes some initial parameter guesses as well as options that are a mutable struct defined on that same julia file (i.e. functions to find the descent direction, prox of non-smooth descent direction function, etc). See IP_alg.jl for more details. There are two main functions that this algorithm is currently dependent upon: Qcustom.jl and DescentMethods.jl. This julia file has FISTA, prox-gradient, and variable projection implementations that one can use to solve for the descent direction in the interior point algorithm. For the simple examples where the non-smooth component $$R_k(s)$$ is zero, one can use minconf_spg. Minconf_spg is a spectral minimization algorithm written by Michael Freidlander and company that is quickly able to project onto the trust-region norm ball one decides to use when solving the descent direction (listed as $$s$$ in both the paper draft and our code). For some norms, one can define the $$l_q$-norm ball projection; for the $$l_1$$-norm ball, projection is done with the OneProjector.jl file, which is currently only for scalar-valued l1 norms (also courtesy of Michael Freidlander and company). The other file, Qcustom.jl, is simply the 2nd order approximation of the objective function; the user supplies the gradient and hessian in another "params" struct. As of right now, these are produced whenever the first objective function is called.

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
