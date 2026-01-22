module RegularizedOptimization

# base dependencies
using LinearAlgebra, Logging, Printf

# external dependencies
using Arpack, ProximalOperators

# dependencies from us
using LinearOperators,
  ManualNLPModels,
  NLPModels,
  NLPModelsModifiers,
  RegularizedProblems,
  ShiftedProximalOperators,
  SolverCore
using Percival: AugLagModel, update_y!, update_μ!

import SolverCore.reset!

const callback_docstring = "
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.xk`: current iterate;
- `solver.∇fk`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.solver_specific[:smooth_obj]`: current value of the smooth part of the objective function;
  - `stats.solver_specific[:nonsmooth_obj]`: current value of the nonsmooth part of the objective function;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything other than `:unknown` will stop the algorithm, but you should use `:user` to properly indicate the intention;
  - `stats.elapsed_time`: elapsed time in seconds.
"

include("utils.jl")
include("input_struct.jl")
include("TR_alg.jl")
include("TRDH_alg.jl")
include("R2_alg.jl")
include("LMModel.jl")
include("LM_alg.jl")
include("LMTR_alg.jl")
include("R2DH.jl")
include("R2NModel.jl")
include("R2N.jl")
include("AL_alg.jl")

end  # module RegularizedOptimization
