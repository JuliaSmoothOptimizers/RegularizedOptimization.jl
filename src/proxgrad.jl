export PG, PGLnsch, PGΔ, PGE

using Printf
"""
    Proximal Gradient Descent  for
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
        feval : number of function evals (total objective )
"""
function PG(GradFcn, Gcn, s, options)

  ε=options.optTol
  max_iter=options.maxIter

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
  ν = options.ν
  s⁺ = deepcopy(s)

  # Iteration set up
  g = GradFcn(s⁺) #objInner/ quadratic model
  feval = 1
  k = 0

  #do iterations
  optimal = false
  tired = k ≥ max_iter

  while !(optimal || tired)

    gold = g
    s = s⁺

    s⁺ = prox(Gcn, s - ν*g, ν) 

    g = GradFcn(s⁺)

    feval+=1
    k+=1
    err = norm(g-gold - (s⁺-s)/ν)
    optimal = err < ε 
    tired = k ≥ max_iter

    k % print_freq == 0 && @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν
 
  end
  return s⁺, feval
end

function PGΔ(GradFcn, Gcn, s, options)

  ε=options.optTol
  max_iter=options.maxIter

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
  ν = options.ν
  p = options.p 
  FcnDec = options.FcnDec

  k = 1
  feval = 1
  s⁺ = deepcopy(s)

  # Iteration set up
  g = GradFcn(s⁺) #objInner/ quadratic model
  @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν

  #do iterations
  optimal = false
  FD = false
  tired = k ≥ max_iter

  while !(optimal || tired || FD)

    gold = g
    s = s⁺

    s⁺ = prox(Gcn, s - ν*g, ν)
    # update function info
    g = GradFcn(s⁺)

    feval+=1
    k+=1
    err = norm(g-gold - (s⁺-s)/ν)
    optimal =  err < ε
    tired = k ≥ max_iter

    k % print_freq == 0 && @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν

    DiffFcn = his[k-1] - his[k] # these do not work 
    FD = abs(DiffFcn)<p*norm(FcnDec)

  end
  return s⁺, feval
end



function PGE(Fcn, Gcn, s, options)

  ε=options.optTol
  max_iter=options.maxIter

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
  ν = options.ν
  p = options.p 
  FcnDec = options.FcnDec

  k = 1
  feval = 1
  s⁺ = deepcopy(s)

  # Iteration set up
  g = GradFcn(s⁺) #objInner/ quadratic model

  #do iterations
  FD = false 
  optimal = false
  tired = k ≥ max_iter 
  
  #do iterations
  while !(optimal || tired || FD)

    gold = g
    s = s⁺

    ν = min(g'*g/(g'*Bk*g), ν) #no BK, will not work 
    #prox step
    s⁺ = prox(Gcn, s - ν*g, ν)
    # update function info
    g = GradFcn(s⁺)

    feval+=1
    k+=1
    err = norm(g-gold - (s⁺-s)/ν)
    optimal =  err < ε
    tired = k ≥ max_iter

    k % print_freq == 0 && @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν

    
    his[k] = f + Gcn(s⁺)*λ # will not work 
    DiffFcn = his[k-1] - his[k]
  end
  return s⁺, his[1:k-1], feval
end

function PGLnsch(GradFcn, Gcn, s, options)

  ε=options.optTol
  max_iter=options.maxIter

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
  p = options.p
  ν₀ = options.ν
  k = 1
  s⁺ = deepcopy(s)

  # Iteration set up
  feval = 1
  g = GradFcn(s⁺) #objInner/ quadratic model

  #do iterations
  optimal = false
  tired = k ≥ max_iter

  while !(optimal || tired)

    gold = g
    s = s⁺

    s⁺ = prox(Gcn, s - ν*g, ν)
    #linesearch but we don't pass in Fcn? 
    while Fcn(s⁺) ≥ f + g'*(s⁺ - s) + 1/(ν*2)*norm(s⁺ - s)^2
        ν *= p*ν
        s⁺ = prox(Gcn, s - ν*g, ν)
        feval+=1
    end
    # update function info
    g = GradFcn(s⁺) #objInner/ quadratic model

    feval+=1
    k+=1
    err = norm(g-gold - (s⁺-s)/ν)
    optimal = err < ε 
    tired = k ≥ max_iter

    k % print_freq == 0 && @info @sprintf "%4d ‖xᵏ⁺¹ - xᵏ‖=%1.5e ν = %1.5e" k err ν
    ν = ν₀

  end
  return s⁺, feval
end