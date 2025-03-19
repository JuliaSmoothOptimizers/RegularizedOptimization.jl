export ROSolverOptions
# TODO remove mk1::Union{Nothing, Function} when done wirh debugging

mutable struct ROSolverOptions{R}
  ϵa::R  # termination criteria
  ϵr::R  # relative stopping tolerance
  neg_tol::R # tolerance when ξ < 0
  Δk::R  # trust region radius
  verbose::Int  # print every so often
  maxIter::Int  # maximum amount of inner iterations
  maxTime::Float64 #maximum time allotted to the algorithm in s
  σmin::R # minimum σk allowed for LM/R2 method
  σk::R # initial σk
  η1::R  # step acceptance threshold
  η2::R  # trust-region increase threshold
  α::R  # νk Δ^{-1} parameter
  ν::R  # initial guess for step length
  γ::R  # trust region buffer
  θ::R  # step length factor in relation to Hessian norm
  β::R  # TR size as factor of first PG step
  reduce_TR::Bool
  dualGap::Union{R, Nothing} # duality gap tolerance for inexact prox computation
  κξ::R # tolerance for the inexact prox computation : \hat{ξ} ≥ κξ * ξ
  callback_pointer::Union{Ptr{Cvoid}, Nothing} # pointer to callback function
  mk1::Union{Nothing, Function} # iR2N's model to give to iR2 for inexact prox computation 

  function ROSolverOptions{R}(;
    ϵa::R = √eps(R),
    ϵr::R = √eps(R),
    neg_tol::R = eps(R)^(1 / 4),
    Δk::R = one(R),
    verbose::Int = 0,
    maxIter::Int = 10000,
    maxTime::Float64 = 3600.0,
    σmin::R = eps(R),
    σk::R = eps(R)^(1 / 5),
    η1::R = √√eps(R),
    η2::R = R(0.9),
    α::R = 1 / eps(R),
    ν::R = eps(R)^(1 / 5),
    γ::R = R(3),
    θ::R = eps(R)^(1 / 5),
    β::R = 1 / eps(R),
    reduce_TR::Bool = true,
    dualGap::Union{R, Nothing} = 1e-5,
    κξ::R = R(3 / 4),
    callback_pointer::Union{Ptr{Cvoid}, Nothing} = @cfunction(
      default_prox_callback,
      Cint,
      (Ptr{Cdouble}, Csize_t, Cdouble, Ptr{Cvoid})
    ),
    mk1::Union{Nothing, Function} = x -> 0.0,
  ) where {R <: Real}
    @assert ϵa ≥ 0
    @assert ϵr ≥ 0
    @assert neg_tol ≥ 0
    @assert Δk > 0
    @assert verbose ≥ 0
    @assert maxIter ≥ 0
    @assert maxTime ≥ 0
    @assert σmin ≥ 0
    @assert σk ≥ 0
    @assert 0 < η1 < η2 < 1
    @assert α > 0
    @assert ν > 0
    @assert γ > 1
    @assert θ > 0
    @assert β ≥ 1
    @assert (isnothing(dualGap) || dualGap ≥ 0)
    @assert κξ > 0
    return new{R}(
      ϵa,
      ϵr,
      neg_tol,
      Δk,
      verbose,
      maxIter,
      maxTime,
      σmin,
      σk,
      η1,
      η2,
      α,
      ν,
      γ,
      θ,
      β,
      reduce_TR,
      dualGap,
      κξ,
      callback_pointer,
      mk1,
    )
  end
end

ROSolverOptions(args...; kwargs...) = ROSolverOptions{Float64}(args...; kwargs...)