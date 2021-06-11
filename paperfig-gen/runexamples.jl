
using TRNC
using Random, LinearAlgebra, Printf, Plots, DataFrames 
using DifferentialEquations, Zygote, DiffEqSensitivity
using ShiftedProximalOperators, ProximalOperators, ProximalAlgorithms
using ADNLPModels, NLPModelsModifiers,NLPModels

pgfplotsx()
include("fig_preproc.jl")
include("tab_preproc.jl")
include("evalwrapper.jl")
include("fig_gen.jl")
include("nonlinfig_gen.jl")
include("modded_panoc.jl")
include("modded_zerofpr.jl")
include("modded_fbs.jl")


Random.seed!(123)

include("runbpdn.jl")
# include("runnonlin.jl")
