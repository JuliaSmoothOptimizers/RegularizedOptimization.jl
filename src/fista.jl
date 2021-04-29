export FISTA, FISTAD
using Printf
"""
  FISTA for
  min_x ϕ(x) = f(x) + g(x), with f(x) cvx and β-smooth, g(x) closed cvx

  Input:
    Fcn: function handle that returns f(x) and ∇f(x)
    Gcn: function handle that returns g(x)
    s: initial point
    proxG: function handle that calculates prox_{νg}
    options: see descentopts.jl
  Output:
    s⁺: s update
    s : s^(k-1)
    his : function history
    feval : number of function evals (total objective)
"""
function FISTA(GradFcn, Gcn, s,  options)
  ε=options.optTol
  max_iter=options.maxIter
  restart = options.restart
  ν = options.ν

  if options.verbose==0
    print_freq = Inf
  elseif options.verbose==1
    print_freq = round(max_iter/10)
  elseif options.verbose==2
    print_freq = round(max_iter/100)
  else
    print_freq = 1
  end

  #Problem Initialize
  y = deepcopy(s)
  s⁺ = zero(s)
  #initialize parameters
  t = 1.0
  # Iteration set up
  k = 0
  feval = 1

  #do iterations
  g = GradFcn(s⁺) #objInner/ quadratic model

  optimal = false
  tired = k ≥ max_iter
  
  while !(optimal || tired)

    copy!(s,s⁺)
    gold = g 
    s⁺ = prox(Gcn, s - ν*g, ν) 

    #update step
    t⁻ = t
    t = 0.5*(1.0 + sqrt(1.0+4.0*t⁻^2))

    #update y
    y .= s⁺ .+ ((t⁻ - 1.0)/t) .* (s⁺.- s)

    g = GradFcn(s⁺)

    feval+=1
    k+=1
    err = norm(g-gold - (s⁺-s)/ν)
    optimal = err < ε 
    tired = k ≥ max_iter
  end
  return s⁺, feval

end

#enforces strict descent  for FISTA 
function FISTAD(GradFcn, Gcn, s, options)
  ε=options.optTol
  max_iter=options.maxIter
  restart = options.restart
  ν = options.ν

  if options.verbose==0
    print_freq = Inf
  elseif options.verbose==1
    print_freq = round(max_iter/10)
  elseif options.verbose==2
    print_freq = round(max_iter/100)
  else
    print_freq = 1
  end

  #Problem Initialize
  v = deepcopy(s)
  s⁺ = zero(s)
  #initialize parameters
  t = 1.0
  # Iteration set up
  k = 0
  feval = 1

  #do iterations
  y = (1.0-t)*s + t*v
  g= GradFcn(y) 

  optimal = false
  tired = k ≥ max_iter
  
  while !(optimal || tired)

    copy!(s,s⁺)
    gold = g 

    #complete prox step 
    u = prox(Gcn, y - ν*g, ν)

    if Fcn(u)[1] ≤ f #this does not work 
      s⁺ = u
    else
      s⁺ = s
    end

    #update step
    # t⁻ = t
    # t = R(0.5)*(R(1.0) + sqrt(R(1.0)+R(4.0)*t⁻^2))
    t = 2/(k + 1)

    #update y
    # v = s⁺ + ((t⁻ - R(1.0))/t)*(s⁺-s)
    v = s⁺ + (1.0/t)*(u - s⁺)
    y = (1.0-t)*s⁺ + t*v #I think this shold be s⁺ since it's at the end of the loop 

    #update parameters
    g = GradFcn(y)

    #check convergence
    err = norm(s - s⁺)
    feval+=1
    k+=1
    optimal = err < ε 
    tired = k ≥ max_iter

  end
  return s⁺, feval

end