export FISTA, FISTAD

"""
  FISTA for
  min_x ϕ(x) = f(x) + g(x), with f(x) cvx and β-smooth, g(x) closed cvx

  Input:
    f: function handle that returns f(x) and ∇f(x)
    h: function handle that returns g(x)
    s: initial point
    proxG: function handle that calculates prox_{νg}
    options: see descentopts.jl
  Output:
    s⁺: s update
    s : s^(k-1)
    his : function history
    feval : number of function evals (total objective)
"""
function FISTA(f, ∇f, h, options; x0::AbstractVector = f.meta.x0)

    ϵ = options.ϵ
    maxIter = options.maxIter

    if options.verbose == 0
        ptf = Inf
    elseif options.verbose == 1
        ptf = round(maxIter / 10)
    elseif options.verbose == 2
        ptf = round(maxIter / 100)
    else
        ptf = 1
    end

    #Problem Initialize
    ν = options.ν
    x = x0
    y = deepcopy(x)
    x⁺ = zero(x)
    Fobj_hist = zeros(maxIter)
    Hobj_hist = zeros(maxIter)
    Complex_hist = zeros(Int, maxIter)

    #initialize parameters
    t = 1.0
    # Iteration set up
    k = 1

    #do iterations
    g = ∇f(x⁺) #objInner/ quadratic model
    fk = f(x⁺)
    hk = h(x⁺)

    optimal = false
    tired = k ≥ maxIter

    if options.verbose != 0
        @info @sprintf "%6s %8s %8s %7s %8s %7s" "iter" "f(x)" "h(x)" "‖∂ϕ‖" "ν" "‖x‖"
    end

    while !(optimal || tired)
        Fobj_hist[k] = fk
        Hobj_hist[k] = hk
        Complex_hist[k] += 1

        copy!(x, x⁺)
        gold = g
        x⁺ = ShiftedProximalOperators.prox(h, x - ν * g, ν)

        #update step
        t⁻ = t
        t = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t⁻^2))

        #update y
        y .= x⁺ .+ ((t⁻ - 1.0) / t) .* (x⁺ .- x)

        g = ∇f(x⁺)
        fk = f(x⁺)
        hk = h(x⁺)

        k += 1
        err = norm(g - gold - (x⁺ - x) / ν)
        optimal = err < ϵ
        tired = k ≥ maxIter

        k % ptf == 0 &&
            @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e " k fk hk err ν norm(xk)

    end
    return x⁺,
    k,
    Fobj_hist[Fobj_hist.!=0],
    Hobj_hist[Fobj_hist.!=0],
    Complex_hist[Complex_hist.!=0]

end

#enforces strict descent  for FISTA
function FISTAD(f, ∇f, h, options; x0::AbstractVector = f.meta.x0)

    ϵ = options.ϵ
    maxIter = options.maxIter


    if options.verbose == 0
        ptf = Inf
    elseif options.verbose == 1
        ptf = round(maxIter / 10)
    elseif options.verbose == 2
        ptf = round(maxIter / 100)
    else
        ptf = 1
    end

    #Problem Initialize
    ν = options.ν
    v = deepcopy(x)
    x⁺ = zero(x)
    Fobj_hist = zeros(maxIter)
    Hobj_hist = zeros(maxIter)
    Complex_hist = zeros(Int, maxIter)

    #initialize parameters
    t = 1.0
    # Iteration set up
    k = 1

    #do iterations
    y = (1.0 - t) * x + t * v
    g = ∇f(y)
    fk = f(y)
    hk = h(y)

    optimal = false
    tired = k ≥ maxIter

    if options.verbose != 0
        @info @sprintf "%6s %8s %8s %7s %8s %7s" "iter" "f(x)" "h(x)" "‖∂ϕ‖" "ν" "‖x‖"
    end

    while !(optimal || tired)

        copy!(x, x⁺)
        gold = g

        #complete prox step
        u = ShiftedProximalOperators.prox(h, y - ν * g, ν)

        if f(u) ≤ f #this does not work
            x⁺ = u
        else
            x⁺ = x
        end

        #update step
        # t⁻ = t
        # t = R(0.5)*(R(1.0) + sqrt(R(1.0)+R(4.0)*t⁻^2))
        t = 2 / (k + 1)

        #update y
        # v = s⁺ + ((t⁻ - R(1.0))/t)*(s⁺-s)
        v = x⁺ + (1.0 / t) * (u - s⁺)
        y = (1.0 - t) * x⁺ + t * v #I think this shold be s⁺ since it's at the end of the loop

        #update parameters
        g = ∇f(y)
        f = f(y)

        #check convergence
        err = norm(g - gold - (x⁺ - x) / ν)
        k += 1
        optimal = err < ϵ
        tired = k ≥ maxIter

        k % ptf == 0 &&
            @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e " k fk hk err ν norm(xk)


    end
    return x⁺,
    k,
    Fobj_hist[Fobj_hist.!=0],
    Hobj_hist[Fobj_hist.!=0],
    Complex_hist[Complex_hist.!=0]
end
