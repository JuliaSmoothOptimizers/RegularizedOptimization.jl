module RegularizedOptimization

# base dependencies
using LinearAlgebra, Logging, Printf

# external dependencies
using ProximalOperators, TSVD

# dependencies from us
using LinearOperators,
  NLPModels, NLPModelsModifiers, RegularizedProblems, ShiftedProximalOperators, SolverCore
using Percival: AugLagModel, update_y!, update_μ!

include("utils.jl")
include("input_struct.jl")
include("PG_alg.jl")
include("Fista_alg.jl")
include("splitting.jl")
include("TR_alg.jl")
include("TRDH_alg.jl")
include("R2_alg.jl")
include("R2DH_alg.jl")
include("R2NModel.jl")
include("R2N_alg.jl")
include("LM_alg.jl")
include("LMTR_alg.jl")
include("AL_alg.jl")

end  # module RegularizedOptimization
