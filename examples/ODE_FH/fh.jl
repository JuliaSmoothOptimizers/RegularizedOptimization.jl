using ADNLPModels, DiffEqSensitivity, DifferentialEquations, NLPModels, ProximalOperators, Random

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
    function misfit_normsq(p)
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

    function misfit(p)
        temp_prob = remake(prob_FH, p=p)
        sol = solve(temp_prob, Vern9(), abstol=1e-14, reltol=1e-14, saveat=savetime)
        if any((sol.retcode != :Success for s in sol))
          @warn "ODE solution failed with parameters" p'
          error("ODE solution failed")
        end
        F = convert(Array, sol)
        F .-= data
        return reshape(F, prod(size(F)), 1)[:]
    end

    return misfit, misfit_normsq
end

# misfit, misfit_normsq = FH_smooth_term()

# model = ADNLPModel(misfit_normsq, ones(5), adbackend=ADNLPModels.ZygoteAD())
# nls_model = ADNLSModel(misfit, ones(5), 202)  # adbackend = ForwardDiff by default
# TODO: don't hardcode the 202

function solve_FH_LM(; λ=1.0, ϵ=1.0e-6)
    misfit, _ = FH_smooth_term()
    nls = ADNLSModel(misfit, ones(5), 202)  # adbackend = ForwardDiff by default
    h = NormL0(λ)
    # TODO: get rid of λ
    inner_options = s_params(1.0, λ; verbose=0)
    params = TRNCmethods(FO_options=inner_options, χ=NormLinf(1.0))
    options = TRNCparams(; maxIter=2000, verbose=10, ϵ=ϵ, β=1e16, σk=1.0e+1)

    xtr, k, Fhist, Hhist, Comp_pg = LM(nls, h, params, options)
    return xtr, k, Fhist, Hhist, Comp_pg
end
