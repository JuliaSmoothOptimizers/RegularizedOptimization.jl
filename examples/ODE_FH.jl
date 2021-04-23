using TRNC
using ADNLPModels
using DiffEqSensitivity, DifferentialEquations, LinearOperators, ForwardDiff
using LinearAlgebra, Random

# function FH_L0Binf()
function ODEFH()
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
    function CostFunc(p)
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

    λ = 1.0
    h = NormL0(λ)

    # put in your initial guesses
    xi = ones(size(pars_FH))
    # this is for l0 norm 
    ϕ = ADNLPModel(CostFunc, xi, adbackend=ADNLPModels.ForwardDiffAD())

    ϵ = 1e-6
    # # set all options
    Doptions = s_params(1.0, λ; optTol = ϵ * (1e-6), verbose=0)
    methods = TRNCmethods(; FO_options=Doptions, s_alg = PGnew, χk=NormLinf(1.0))
    params = TRNCparams(; maxIter=500, verbose=10, ϵ=ϵ, β=1e16)

    xtr, k, Fhist, Hhist, Comp_pg = TR(ϕ, h, methods, params)

end
    # Ψ.ψχprox = proxl0s
    # parametersLM = TRNCstruct(ϕ, Ψ; FO_options=Doptions, χk=(s) -> norm(s, Inf))
    # optionsLM = TRNCoptions(; σk=1e4, ϵ=ϵ, verbose=10) # options, such as printing (same as above), tolerance, γ, σ, τ, w/e

    # input initial guess
    # xlm, klm, Fhistlm, Hhistlm, Comp_pglm = QRalg(xi, parametersLM, optionsLM)


    # @show xtr
    # @show xlm
    # @show x0



    # function h_obj(x)
    #     if norm(x,0) ≤ δ
    #         h = 0
    #     else
    #         h = Inf
    #     end
    #     return h 
    # end

    # function prox(q, σ, xk, Δ)
    #     ProjB(w) = min.(max.(w, xk.-Δ), xk.+Δ)
    #     y = q + xk 
    #     #find largest entries
    #     p = sortperm(abs.(y), rev = true)
    #     y[p[δ+1:end]].=0 #set smallest to zero 
    #     y = ProjB(y)#put largest entries in projection
    #     s = y - xk 

    #     return s 
# end

# function solve_FH_trust_region(; λ=1.0, ϵ=1.0e-6)
#     ϕ = FH_smooth_term()
#     Doptions = s_params(1.0, λ; optTol=ϵ * (1e-6), verbose=0)
#     params = TRNCmethods(ϕ,
#                         #  (args...; kwargs...) -> error("should not have called this method!"),
#                          zero_norm;
#                          FO_options=Doptions,
#                          ψχprox=zero_norm_prox_trust_region,
#                          χk=s -> norm(s, Inf)
#                          )

#     options = TRNCparams(; maxIter=500, verbose=10, ϵ=ϵ, β=1e16)

#     # input initial guess
#     xi = ones(5)
#     xtr, k, Fhist, Hhist, Comp_pg = TR(xi, params, options)
#     return xtr, k, Fhist, Hhist, Comp_pg
# end

# function solve_FH_QR(; λ=1.0, ϵ=1.0e-4)
#     # f_obj = FH_smooth_term()
#     ϕ = FH_smooth_term()
#     Doptions = s_params(1.0, λ; optTol=ϵ * (1e-6), verbose=0)
#     h = NormL0(5, λ)
#     parametersQR = TRNCmethods(ϕ,
#                                h;
#                                FO_options=Doptions,
#                                ψχprox=(args...) -> prox(h, args...),  # zero_norm_prox,
#                                χk=s -> norm(s, Inf))
 
#     optionsQR = TRNCparams(; maxIter=10000, σk=1e4, ϵD=ϵ, verbose=2) # options, such as printing (same as above), tolerance, γ, σ, τ, w/e

#     # input initial guess
#     xi = ones(5)
#     xQR, kQR, FhistQR, HhistQR, Comp_pgQR = TRNC.QRalg(xi, parametersQR, optionsQR)
#     return xQR, kQR, FhistQR, HhistQR, Comp_pgQR
# end
