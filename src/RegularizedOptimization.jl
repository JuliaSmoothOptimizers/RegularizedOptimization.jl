module RegularizedOptimization

# base dependencies
using LinearAlgebra, Logging, Printf

# external dependencies
<<<<<<< HEAD
using ProximalOperators, TSVD
=======
using Arpack, ProximalOperators, LinearOperators, Krylov
>>>>>>> 6b06143 (committing changes for demo-fh to work)

# dependencies from us
using LinearOperators, NLPModels, NLPModelsModifiers, ShiftedProximalOperators, SolverCore

include("utils.jl")
include("input_struct.jl")
include("PG_alg.jl")
include("Fista_alg.jl")
include("RS_alg.jl")
include("TR_alg.jl")
include("TRDH_alg.jl")
include("R2_alg.jl")
include("LM_alg.jl")
include("LMTR_alg.jl")

end  # module RegularizedOptimization
