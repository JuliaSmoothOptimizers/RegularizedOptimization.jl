using Random, LinearAlgebra, TRNC 
using ProximalOperators, ShiftedProximalOperators 
using NLPModels, NLPModelsModifiers, ADNLPModels

# min_x 1/2||Ax - b||^2 + λ||x||₀; ΔB_∞
function L0BInf()
  compound = 1

  m,n = compound*200,compound*512 #if you want to rapidly change problem size 
  k = compound*10 #10 signals 
  α = .01 #noise level 

  #start bpdn stuff 
  x0 = zeros(n)
  p   = randperm(n)[1:k]
  x0 = zeros(n,)
  x0[p[1:k]]=sign.(randn(k)) #create sparse signal 

  A,_ = qr(randn(n,m))
  B = Array(A)'
  A = Array(B)

  b0 = A*x0
  b = b0 + α*randn(m,)

  #put in your initial guesses
  xi = zeros(n,)
  λ = norm(A'*b, Inf)/10 #this can change around 

  ϕ = LSR1Model(SmoothObj((x) -> .5*norm(A*x - b)^2, (x) -> A'*(A*x - b), xi))
  # ϕ = ADNLPModel((x) -> .5*norm(A*x - b)^2, xi)

  h = NormL0(λ)

  #set options for inner algorithm - only requires ||Bk|| norm guess to start (and λ but that is updated in TR)
  #verbosity is levels: 0 = nothing, 1 -> maxIter % 10, 2 = maxIter % 100, 3+ -> print all 
  β = opnorm(A)^2 #1/||Bk|| for exact Bk = A'*A
  Doptions=s_params(1/β, λ; verbose=0, optTol=1e-16)


  ε = 1e-6
  #define parameters - must feed in smooth, nonsmooth, and λ
  #first order options default ||Bk|| = 1.0, no printing. PG is default inner, Rkprox is inner prox loop - defaults to 2-norm ball projection (not accurate if h=0)
  parameters = TRNCmethods(; FO_options = Doptions, s_alg=PGnew, χk=NormLinf(1.0))
  options = TRNCparams(; ϵ=ε, verbose = 10, θ = 1e-3, Δk = 1.0) #options, such as printing (same as above), tolerance, γ, σ, τ, w/e


  #input NLP, h, parameters, options 
  xtr, k, Fhist, Hhist, Comp_pg = TR(ϕ, h, parameters, options)

  optionsQR = TRNCparams(; σk = 1/β, ϵ=ε, verbose = 10) #options, such as printing (same as above), tolerance, γ, σ, τ, w/e
  xi .= 0 
  #input initial guess
  xqr, kqr, Fhistqr, Hhistqr, Comp_pgqr = QRalg(ϕ, h, parameters, optionsQR)

  @info "TR relative error" norm(xtr - x0) / norm(x0)
  @info "QR relative error" norm(xqr - x0) / norm(x0)
  @info "monotonicity" findall(>(0), diff(Fhist+Hhist))

end
