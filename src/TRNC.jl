module TRNC

# base dependencies
using LinearAlgebra, Printf, Random

# external dependencies
using Arpack, ProximalOperators, Logging

# dependencies from us
using ADNLPModels, NLPModels, NLPModelsModifiers, ShiftedProximalOperators

include("descentopts.jl")
include("PG_alg.jl")
include("Fista_alg.jl")
include("splitting.jl")
include("TR_alg.jl")
include("QR_alg.jl")
include("LM_alg.jl")
include("LMTR_alg.jl")

end  # module TRNC
