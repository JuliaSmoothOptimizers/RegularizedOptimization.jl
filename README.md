# Julia Framework for Interior-Point Trust-Region Method for Composite Optimization


## Overview
This folder contains code to run IP TR methods for non-smooth (potentially) nonconvex composite optimization problems. Currently, only Algorithm 4.2 (Non-smooth Trust Region) is implemented as IntPt_TR in the file "IP_alg.jl". To run an easy example, simply type 
```julia
julia IPscript.jl
```
to solve the problem $$\min_x \ \|Ax - b\|^2$$ for randomly generated 100x100 matrix $$A$$ and gaussian noise observations $$b$$. This script allows the user to define the function to solve in the same script and pass said function to IntPt_TR, which takes in that function, some initial parameter guesses for $$x, zl, zu$$, and a bank of options for IntPt_TR. This seems to do about the same as the Matlab code it was adapted from. Make sure you have Julia 1.0+. 

## Algorithm 4.2 - Nonsmooth Trust Region
In the draft, algorithm 4.2 is represented by IntPt_TR, a function that is on the julia file IP_alg.jl. While more about the function is described in the aforementioned file, it basically takes some initial parameter guesses as well as options that are a mutable struct defined on that same julia file. See IP_alg.jl for more details. There are two main functions that this algorithm is currently dependent upon: Qcustom.jl and minconf_spg. Minconf_spg is a spectral (?) minimization algorithm written by Michael Freidlander and company for which we are currently using to solve for our descent direction (listed as $$s$$ in both the paper draft and our code) on the $$\|\cdot\|\_1$$ norm  ball. This projection is done with the OneProjector.jl file, which is currently only for scalar-valued l1 norms (also courtesy of Michael Freidlander and company). The other file, Qcustom.jl, is simply the 2nd order approximation of the objective function; the user supplies the gradient and hessian in another "params" struct. As of right now, these are produced whenever the first objective function is called. 

In summary, the code structure is: 
				Calls       |      Defines
				--------------------------
- IPscript.jl:  IntPt_TR	  objective
				IP_params

- IP_alg.jl:    minconf_spg   IntPt_TR
				OneProjector  IP_params
				Qcustom

- minconf_spg/: OneProjector  minconf_spg
				barrier obj   spg_params

- Qcustom.jl:   grad, Hess    QCustom
							  Q_params


## To do: 
1) Insert ability to call different methods to solve for the descent direction
2) Make Algorithm 4.3 as a wrapper
3) Come up with a better example? 