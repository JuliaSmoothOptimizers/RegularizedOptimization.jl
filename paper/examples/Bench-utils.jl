module BenchUtils

using ProximalAlgorithms
using ProximalCore
using NLPModels

export Counting, reset_counters!, make_adnlp_compatible!

(f::AbstractNLPModel)(x) = obj(f,x)

function ProximalAlgorithms.value_and_gradient(f::AbstractNLPModel, x)
    return obj(f,x), grad(f, x)
end

"Wrapper compteur pour f ou g (compte #obj, #∇f, #prox)."
mutable struct Counting{T}
    f::T
    eval_count::Int
    gradient_count::Int
    prox_count::Int
end
Counting(f::T) where {T} = Counting{T}(f, 0, 0, 0)

# f(x)
(f::Counting)(x) = (f.eval_count += 1; f.f(x))

# (f, ∇f)
function ProximalAlgorithms.value_and_gradient(f::Counting, x)
    f.eval_count += 1
    f.gradient_count += 1
    return ProximalAlgorithms.value_and_gradient(f.f, x)
end

# prox!(y, g, x, γ)
function ProximalCore.prox!(y, g::Counting, x, γ)
    g.prox_count += 1
    return ProximalCore.prox!(y, g.f, x, γ)
end

"Réinitialise les compteurs d’un Counting."
reset_counters!(c::Counting) = (c.eval_count = 0; c.gradient_count = 0; c.prox_count = 0; nothing)

end # module
