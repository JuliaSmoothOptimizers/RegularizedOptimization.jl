using ADNLPModels, DifferentialEquations, NLPModels, NLPModelsModifiers, ProximalOperators, Random, LinearAlgebra

function FH_smooth_term()
    function FH_ODE(dx, x, p, t)
        V, W = x
        I, μ, a, b, c = p
        dx[1] = (V - V^3 / 3 - W + I) / μ
        dx[2] = μ * (a * V - b * W + c)
    end

    u0 = [2.0; 0.0]
    tspan = (0.0, 20.0)
    savetime = 0.2

    pars_FH = [0.5, 0.08, 1.0, 0.8, 0.7]
    prob_FH = ODEProblem(FH_ODE, u0, tspan, pars_FH)

    # FH model = van-der-Pol oscillator when I = b = c = 0
    # x' = μ(x - x^3/3 - y)
    # y' = x/μ -> here μ = 12.5
    # changing the parameters to p = [0, .08, 1.0, 0, 0]
    x0 = [0, 0.2, 1.0, 0, 0]
    prob_VDP = ODEProblem(FH_ODE, u0, tspan, x0)
    sol_VDP = solve(prob_VDP, reltol = 1e-6, saveat = savetime)

    # add random noise to vdP solution
    t = sol_VDP.t
    b = hcat(sol_VDP.u...)
    noise = 0.1 * randn(size(b))
    data = noise + b

    # solve FH with parameters p
    function simulate(p)
        temp_prob = remake(prob_FH, p = p)
        sol = solve(temp_prob, Vern9(), abstol = 1e-14, reltol = 1e-14, saveat = savetime)
        if any((sol.retcode != :Success for s in sol))
            @warn "ODE solution failed with parameters" p'
            error("ODE solution failed")
        end
        F = convert(Array, sol)
        return F
    end

    # define residual vector
    function residual(p)
        F = simulate(p)
        F .-= data
        return reshape(F, prod(size(F)), 1)[:]
    end

    # misfit = ‖residual‖² / 2
    function misfit(p)
        F = residual(p)
        return dot(F, F) / 2
    end

    return data, simulate, residual, misfit
end

function solve_FH_LM(; λ = 1.0, ϵ = 1.0e-6)
    data, simulate, resid, misfit = FH_smooth_term()
    nls = ADNLSModel(resid, ones(5), 202)  # adbackend = ForwardDiff by default
    h = NormL0(λ)
    parameters = TRNCoptions(; ν = 1.0e-1, β = 1e16, ϵ = ϵ, verbose = 10)
    χ = NormLinf(1.0)

    xtr, k, Fhist, Hhist, Comp_pg = LM(nls, h, χ, parameters)
    return xtr, k, Fhist, Hhist, Comp_pg
end

function solve_FH_TR(; λ = 1.0, ϵ = 1.0e-6)
    data, simulate, resid, misfit = FH_smooth_term()
    nlp = LBFGSModel(ADNLPModel(misfit, ones(5))) # adbackend = ForwardDiff by default
    h = NormL0(λ)
    parameters = TRNCoptions(; β = 1e16, ϵ = ϵ, verbose = 10)
    χ = NormLinf(1.0)

    # standard logging
    xtr, k, Fhist, Hhist, Comp_pg = TR(nlp, h, χ, parameters)

    return xtr, k, Fhist, Hhist, Comp_pg
end

