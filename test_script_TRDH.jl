using LinearAlgebra, NLPModels, NLPModelsModifiers, RegularizedProblems, ProximalOperators, Random, ShiftedProximalOperators
Random.seed!(0)
bpdn = LSR1Model(bpdn_model(1, bounds = true)[1])
λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10
h = NormL0(λ)
χ = NormLinf(1.0)  

#import Pkg; Pkg.add(url = "https://github.com/MaxenceGollier/RegularizedOptimization.jl.git", rev = "TR-JSO")
using RegularizedOptimization
options = ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10)

TR(bpdn, h, χ, options, subsolver = TRDHSolver)





error("done")
using RegularizedOptimization, RegularizedProblems, NLPModels, NLPModelsModifiers, LinearAlgebra, ProximalOperators, Random
Random.seed!(1234)

compound = 1 
bpdn, _, _ = bpdn_model(compound)
options = ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10)
λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10
x0 = zeros(bpdn.meta.nvar)
χ = NormLinf(1.0)
h  = NormL1(λ)
nlp = SpectralGradientModel(bpdn)

out = TRDH(nlp, h, χ, options, x0 = x0)

reg_nlp = RegularizedNLPModel(nlp, h)
solver = TRDHSolver(reg_nlp, χ = χ)
stats = RegularizedExecutionStats(reg_nlp)
solve!(solver, reg_nlp, stats, x=x0, verbose = 1)

macro wrappedallocs(expr)
    kwargs = [a for a in expr.args if isa(a, Expr)]
    args = [a for a in expr.args if isa(a, Symbol)]
  
    argnames = [gensym() for a in args]
    kwargs_dict = Dict{Symbol, Any}(a.args[1] => a.args[2] for a in kwargs if a.head == :kw)
    quote
      function g($(argnames...); kwargs_dict...)
        @allocated $(Expr(expr.head, argnames..., kwargs...))
      end
      $(Expr(:call, :g, [esc(a) for a in args]...))
    end
end

@wrappedallocs solve!(solver, reg_nlp, stats)


