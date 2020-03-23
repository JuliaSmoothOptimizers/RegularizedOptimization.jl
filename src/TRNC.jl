module TRNC
using LinearAlgebra, Printf, Roots, Random

export TRNCPATH
TRNCPATH = dirname(@__DIR__)

include("minconf_spg/SLIM_optim.jl")
include("barrier.jl")
include("DescentMethods.jl")
include("IP_alg.jl")
include("ProxLQ.jl")
include("ProxProj.jl")
include("Qcustom.jl")
include("ProxBinf.jl")
include("ProxB2.jl")


end  # module TRNC
