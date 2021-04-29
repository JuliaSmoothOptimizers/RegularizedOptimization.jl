using Random, LinearAlgebra, TRNC 
using ProximalOperators, ShiftedProximalOperators 
using NLPModels, NLPModelsModifiers, ADNLPModels

# min_x 1/2||Ax - b||^2 + λ||x||₀; ΔB_∞
function B0Binf(compound=1)
  m, n = compound * 200, compound * 512 # if you want to rapidly change problem size 
  k = compound * 10 # 10 signals 
  α = .01 # noise level 

  # start bpdn stuff 
  x0 = zeros(n)
  p   = randperm(n)[1:k]
  x0 = zeros(n, )
  x0[p[1:k]] = sign.(randn(k)) # create sparse signal 

  A, _ = qr(randn(n, m))
  B = Array(A)'
  A = Array(B)

  b0 = A * x0
  b = b0 + α * randn(m, )

  λ = 1.0 
  # put in your initial guesses
  xi = zeros(n, )
  # set options for inner algorithm - only requires ||Bk|| norm guess to start (and λ but that is updated in TR)
  # verbosity is levels: 0 = nothing, 1 -> maxIter % 10, 2 = maxIter % 100, 3+ -> print all 
  β = opnorm(A)^2 # 1/||Bk|| for exact Bk = A'*A
  Doptions = s_params(1 / β, λ; verbose=0, optTol=1e-16)

  function gradF!(g,x)
      g .= A'*(A*x - b)
      return g
    end
  ϕ = LSR1Model(SmoothObj((x) -> .5*norm(A*x - b)^2, gradF!, xi))
  h = IndBallL0(k)

  ϵ = 1e-6
  methods = TRNCmethods(; FO_options=Doptions, s_alg=PG, χ=NormLinf(1.0))
  parameters = TRNCparams(;β = 1e16, ϵ=ϵ, verbose = 10)

  # input initial guess, parameters, options 
  x_pr, k, Fhist, Hhist, Comp_pg = TR(ϕ, h, methods, parameters)
  # final value, kth iteration, smooth history, nonsmooth history (with λ), # of evaluations in the inner PG loop 

  paramsQR = TRNCparams(; σk=1 / β, ϵ=ϵ, verbose=10) # options, such as printing (same as above), tolerance, γ, σ, τ, w/e

  # input initial guess
  xqr, kqr, Fhistqr, Hhistqr, Comp_pgqr = QRalg(ϕ, h, methods, paramsQR)

  @info "TR relative error" norm(x_pr - x0) / norm(x0)
  @info "QR relative error" norm(xqr - x0) / norm(x0)
  @info "monotonicity" findall(>(0), diff(Fhist + Hhist))
end