module TRNC

# base dependencies
using LinearAlgebra, Printf, Random

# external dependencies
using Arpack, ProximalOperators

# dependencies from us
using ADNLPModels, NLPModels, NLPModelsModifiers, ShiftedProximalOperators

include("descentopts.jl")
include("proxgrad.jl")
include("fista.jl")
include("splitting.jl")
include("linesearch.jl")
include("TR_alg.jl")
include("QR_alg.jl")
include("LM_alg.jl")
include("LMTR_alg.jl")

end  # module TRNC
