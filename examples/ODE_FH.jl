using TRNC
using DiffEqSensitivity, DifferentialEquations, LinearOperators, Roots, Zygote
using LinearAlgebra, Printf, Random

# function FH_L0Binf()
function FH_smooth_term()
    # so we need a model solution, a gradient, and a Hessian of the system (along with some data to fit)
    function FH_ODE(dx, x, p, t)
        # p is parameter vector [I,μ, a, b, c]
        V, W = x 
        I, μ, a, b, c = p
        dx[1] = (V - V^3 / 3 -  W + I) / μ
        dx[2] = μ * (a * V - b * W + c)
    end

    u0 = [2.0; 0.0]
    tspan = (0.0, 20.0)
    savetime = .2

    pars_FH = [0.5, 0.08, 1.0, 0.8, 0.7]
    prob_FH = ODEProblem(FH_ODE, u0, tspan, pars_FH)

    # So this is all well and good, but we need a cost function and some parameters to fit. First, we take care of the parameters
    # We start by noting that the FHN model is actually the van-der-pol oscillator with some parameters set to zero
    # x' = μ(x - x^3/3 - y)
    # y' = x/μ -> here μ = 12.5
    # changing the parameters to p = [0, .08, 1.0, 0, 0]
    x0 = [0, .2, 1.0, 0, 0]
    prob_VDP = ODEProblem(FH_ODE, u0, tspan, x0)
    sol_VDP = solve(prob_VDP, reltol=1e-6, saveat=savetime)

    # also make some noise to fit later
    t = sol_VDP.t
    b = hcat(sol_VDP.u...)
    noise = .1 * randn(size(b))
    data = noise + b

    # so now that we have data, we want to formulate our optimization problem. This is going to be 
    # min_p ||f(p) - b||₂^2 + λ||p||₀
    # define your smooth objective function
    function Gradprob(p)
        temp_prob = remake(prob_FH, p=p)
        temp_sol = solve(temp_prob, Vern9(), abstol=1e-14, reltol=1e-14, saveat=savetime)
        tot_loss = 0.

        if any((temp_sol.retcode != :Success for s in temp_sol))
            tot_loss = Inf
        else
            temp_v = convert(Array, temp_sol)
            tot_loss = sum(((temp_v - data).^2) ./ 2)
        end
        return tot_loss
    end

    function f_obj(x) # gradient and hessian info are smooth parts, m also includes nonsmooth part
        fk = Gradprob(x)
        if fk == Inf 
            grad = zeros(size(x))
            # Hess = Inf*ones(size(x,1), size(x,1))
        else
            grad = Zygote.gradient(Gradprob, x)[1] 
            # Hess = Zygote.hessian(Gradprob, x)
        end
        return fk, grad
    end

    f_grad(x) = Zygote.gradient(Gradprob, x)[1]

    # return f_obj
    return Gradprob, f_grad
end

function zero_norm(x)
    return norm(x, 0) 
end

function zero_norm_prox_trust_region(q, σ, xk, Δ)
    ProjB(y) = min.(max.(y, xk .- Δ), xk .+ Δ) # define outside? 
    c = sqrt(2 * σ)
    w = xk + q
    st = zeros(size(w))

    for i = 1:length(w)
        absx = abs(w[i])
        if absx <= c
            st[i] = 0
        else
            st[i] = w[i]
        end
    end
    s = ProjB(st) - xk
    return s 
end

mutable struct NormL0
    w
    st
    function NormL0(n)
        new(Vector{Float64}(undef, n), Vector{Float64}(undef, n))
    end
end

(h::NormL0)(x) = norm(x, 0)

function prox(h::NormL0, q, σ, xk)
    c = sqrt(2 * σ)
    h.w .= xk .+ q

    for i = 1:length(h.w)
        absx = abs(h.w[i])
        if absx ≤ c
            h.st[i] = 0
        else
            h.st[i] = h.w[i]
        end
    end
    h.st .-= xk
    return h.st
end

function prox(h::NormL0, q, σ, xk, Δ)
    ProjB!(y) = begin
        for i ∈ eachindex(y)
            y[i] = min(max(y[i], xk[i] - Δ), xk[i] + Δ)
        end
    end
    c = sqrt(2 * σ)
    h.w .= xk .+ q

    for i = 1:length(h.w)
        absx = abs(h.w[i])
        if absx ≤ c
            h.st[i] = 0
        else
            h.st[i] = h.w[i]
        end
    end
    ProjB!(h.st)
    h.st .-= xk
    return h.st
end

function zero_norm_prox(q, σ, xk, args...)
    c = sqrt(2 * σ)
    w = xk + q
    st = zeros(length(w))

    for i = 1:length(w)
        absx = abs(w[i])
        if absx ≤ c
            st[i] = 0
        else
            st[i] = w[i]
        end
    end
    s = st - xk
    return s
end

function solve_FH_trust_region(; λ=1.0, ϵ=1.0e-6)
    f_obj, f_grad = FH_smooth_term()
    Doptions = s_params(1.0; λ=λ, optTol=ϵ * (1e-6), verbose=0)
    params = TRNCmethods(x -> (f_obj(x), f_grad(x)),
                         (args...; kwargs...) -> error("should not have called this method!"),
                         zero_norm,
                         λ;
                         FO_options=Doptions,
                         ψχprox=zero_norm_prox_trust_region,
                         χk=s -> norm(s, Inf),
                         HessApprox=LSR1Operator)

    options = TRNCparams(; maxIter=500, verbose=10, ϵD=ϵ, β=1e16)

    # input initial guess
    xi = ones(5)
    xtr, k, Fhist, Hhist, Comp_pg = TR(xi, params, options)
    return xtr, k, Fhist, Hhist, Comp_pg
end

function solve_FH_QR(; λ=1.0, ϵ=1.0e-4)
    # f_obj = FH_smooth_term()
    f_obj, f_grad = FH_smooth_term()
    Doptions = s_params(1.0; λ=λ, optTol=ϵ * (1e-6), verbose=0)
    h = NormL0(5)
    parametersQR = TRNCmethods(f_obj,
                               f_grad,
                               h,  # zero_norm,
                               λ;
                               FO_options=Doptions,
                               ψχprox=(args...) -> prox(h, args...),  # zero_norm_prox,
                               χk=s -> norm(s, Inf))
 
    optionsQR = TRNCparams(; maxIter=10000, σk=1e4, ϵD=ϵ, verbose=2) # options, such as printing (same as above), tolerance, γ, σ, τ, w/e

    # input initial guess
    xi = ones(5)
    xQR, kQR, FhistQR, HhistQR, Comp_pgQR = TRNC.QR(xi, parametersQR, optionsQR)
    return xQR, kQR, FhistQR, HhistQR, Comp_pgQR
end

function zero_norm_ball(x, δ=1)
    if norm(x, 0) ≤ δ
        h = 0.0
    else
        h = Inf
    end
    return h 
end

function zero_norm_ball_prox_trust_region(q, σ, xk, Δ, δ=1)
    ProjB(w) = min.(max.(w, xk.-Δ), xk.+Δ)
    y = q + xk 
    #find largest entries
    p = sortperm(abs.(y), rev = true)
    y[p[δ+1:end]].=0 #set smallest to zero 
    y = ProjB(y)#put largest entries in projection
    s = y - xk 
    return s 
end
