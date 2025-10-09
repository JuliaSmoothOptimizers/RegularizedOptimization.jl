using Pkg; Pkg.activate(".")
using LinearAlgebra
using RegularizedOptimization
using RegularizedProblems
using NLPModels
using ProximalOperators

# Minimal quadratic model: f(x) = 1/2 x'Ax - b'x
n = 50
A = diagm(0 => fill(2.0, n))
b = ones(n)

# Create a simple NLPModels-compatible model via ManualNLPModels or a QuadraticModel wrapper.
# Here we use QuadraticModel from QuadraticModels.jl to construct an NLPModel and wrap it.
using QuadraticModels
using LinearOperators

c = -b
H = LinearOperators.opEye(Float64, n) * 2.0
qm = QuadraticModel(c, H, c0 = 0.0, x0 = zeros(n), name = "test-quad")
reg_nlp = RegularizedNLPModel(qm, NormL2(0.0))
stats = RegularizedExecutionStats(reg_nlp)

# Build R2NSolver
solver = R2NSolver(reg_nlp)
stats = RegularizedExecutionStats(reg_nlp)

# Warm up call (to compile)
try
  solve!(solver, reg_nlp, stats; σk = 1.0, atol = 1e-6, rtol = 1e-6)
catch err
  # ignore any runtime errors during warmup
  @warn "warmup solve! failed" err
end

println("Measuring allocations for solve! at initial call")
alloc = @allocated begin
  stats = solve!(solver, reg_nlp, stats; σk = 1.0, atol = 1e-6, rtol = 1e-6)
end
println("Allocations: ", alloc)
println("Status: ", stats.status)
