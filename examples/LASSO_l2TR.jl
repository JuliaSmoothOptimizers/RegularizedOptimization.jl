using Random, LinearAlgebra, TRNC 
using ProximalOperators, ShiftedProximalOperators 
using NLPModels, NLPModelsModifiers, ADNLPModels

# min_x 1/2||Ax - b||^2 + λ||x||₁; ΔB_1
function L1B2(compound=1)
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

  λ = norm(A' * b, Inf) / 10 # this can change around 

  # define your smooth objective function
  xi = zeros(size(x0))
  function grad!(g, x)
    g .= A'*(A*x - b)
    return g
  end
  ϕ = LSR1Model(SmoothObj((x) -> .5*norm(A*x - b)^2, grad!, xi))

  h = NormL1(λ)

  β = opnorm(A)^2 #1/||Bk|| for exact Bk = A'*A
  Doptions=s_params(1/β, λ; verbose=0, optTol=1e-16)

  ϵ = 1e-6
  methods = TRNCmethods(; FO_options=Doptions, s_alg=PGnew, χ=NormL2(1.0))
  parameters = TRNCparams(;β = 1e16, ϵ=ϵ, verbose = 10)

  # input initial guess, parameters, options 
  xtr, ktr, Fhisttr, Hhisttr, Comp_pgtr = TR(ϕ, h, methods, parameters)


  # input initial guess, parameters, options 
  paramsQR = TRNCparams(; σk = 1/β, ϵ=ϵ, verbose = 10)
  xi .= 0 

  xqr, kqr, Fhistqr, Hhistqr, Comp_pgqr = QRalg(ϕ, h, methods, paramsQR)


  @info "TR relative error" norm(xtr - x0) / norm(x0)
  @info "QR relative error" norm(xqr - x0) / norm(x0)
  @info "monotonicity" findall(>(0), diff(Fhisttr + Hhisttr))

end