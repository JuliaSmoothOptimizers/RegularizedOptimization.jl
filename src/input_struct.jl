export ROSolverOptions

mutable struct ROSolverOptions{R}
  ϵ :: R  # termination criteria
  Δk :: R  # trust region radius
  verbose :: Int  # print every so often
  maxIter :: Int  # maximum amount of inner iterations
  maxTime :: R #maximum time allotted to the algorithm in s
  η1 :: R  # step acceptance threshold
  η2 :: R  # trust-region increase threshold
  α :: R  # νk Δ^{-1} parameter
  ν :: R  # initial guess for step length
  γ :: R  # trust region buffer
  θ :: R  # step length factor in relation to Hessian norm
  β :: R  # TR size as factor of first PG step

  function ROSolverOptions{R}(
    ;
    ϵ :: R = √eps(R),
    Δk :: R = one(R),
    verbose :: Int = 0,
    maxIter :: Int = 10000,
    maxTime :: R = R(10000),
    η1 :: R = √√eps(R),
    η2 :: R = R(0.9),
    α :: R = 1 / eps(R),
    ν :: R = 1.0e-3,
    γ :: R = R(3.0),
    θ :: R = R(1e-3),
    β :: R = R(10.0),
    ) where {R <: Real}
    return new{R}(ϵ, Δk, verbose, maxIter, maxTime, η1, η2, α, ν, γ, θ, β)
  end
end

ROSolverOptions(args...; kwargs...) = ROSolverOptions{Float64}(args...; kwargs...)

