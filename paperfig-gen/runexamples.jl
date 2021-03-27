
using TRNC
using Random, LinearAlgebra,LinearOperators, Printf,Roots, Plots, DataFrames 
using DifferentialEquations, Zygote, DiffEqSensitivity
using ProximalOperators, ProximalAlgorithms

pgfplotsx()
include("fig_preproc.jl")
include("tab_preproc.jl")
include("evalwrapper.jl")
include("fig_gen.jl")
include("nonlinfig_gen.jl")
include("modded_panoc.jl")
include("modded_zerofpr.jl")
include("modded_fbs.jl")


Random.seed!(0)

# include("runbpdn.jl")
include("runnonlin.jl")
