module TRNC
using LinearAlgebra, Printf, Roots, Random

export TRNCPATH
TRNCPATH = dirname(@__DIR__)

include("descentopts.jl")
include("proxgrad.jl")
include("fista.jl")
include("splitting.jl")
include("linesearch.jl")
include("Derivatives.jl")
include("PowerIter.jl")
include("IP_alg.jl")
include("proxGD.jl")


end  # module TRNC
