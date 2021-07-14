using Random, LinearAlgebra, TRNC 
using ProximalOperators, ShiftedProximalOperators 
using NLPModels, NLPModelsModifiers, ADNLPModels
Random.seed!(1234)
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

  function gradF!(g,x)
    g .= A'*(A*x - b)
    return g
  end

  ϕ = LSR1Model(SmoothObj((x) -> .5*norm(A*x - b)^2, gradF!, xi))
  h = NormL0(λ)
  χ=NormLinf(1.0)

  ν = opnorm(A)^2 #1/||Bk|| for exact Bk = A'*A

  ϵ = 1e-6 
  parameters = TRNCoptions(;ν = 1.0,  β = 1e16, ϵ=ϵ, verbose = 10)

  #input NLP, h, parameters, options 
  xtr, k, Fhist, Hhist, Comp_pg = TR(ϕ, h, χ, parameters; s_alg = QRalg)
  # x_pr, k, Fhist, Hhist, Comp_pg = LMTR(ϕ, h, methods, parameters)

  xi .= 0 
  reset!(ϕ)
  
  #input initial guess
  xqr, kqr, Fhistqr, Hhistqr, Comp_pgqr = QRalg(ϕ, h, parameters; x0 = xi)

  @info "TR relative error" norm(xtr - x0) / norm(x0)
  @info "QR relative error" norm(xqr - x0) / norm(x0)
  @info "monotonicity" findall(>(0), diff(Fhist+Hhist))

end
