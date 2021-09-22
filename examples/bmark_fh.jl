include("LMTR_TR-FitzHugh_datagen.jl")
using BenchmarkTools
using DiffEqSensitivity  # for reverse AD in DifferentialEquations
using ReverseDiff  # for ADNLPModels

function bmark(model)
    x = rand(model.meta.nvar)
    u = ones(model.meta.nvar)
    v = zeros(model.nls_meta.nequ)
    out = @benchmark jprod_residual!($model, $x, $u, $v)
    @info "jprod_residual!" out
    out = @benchmark jtprod_residual!($model, $x, $v, $u)
    @info "jtprod_residual!" out
    J = jac_op_residual(model, x)
    out = @benchmark mul!($v, $J, $u)
    @info "op * u" out
    out = @benchmark mul!($u, $J', $v)
    @info "op' * v" out
end

data, simulate, resid, misfit = FH_smooth_term()

@info "ForwardDiff model"
nls_fd = ADNLSModel(resid, ones(5), 202)  # adbackend = ForwardDiff by default
bmark(nls_fd)

@info "ReverseDiff model"
nls_rd = ADNLSModel(resid, ones(5), 202, adbackend = ADNLPModels.ReverseDiffAD(5, 202))
bmark(nls_rd)
