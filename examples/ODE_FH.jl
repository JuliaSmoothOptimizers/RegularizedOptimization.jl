using DifferentialEquations, Zygote, DiffEqSensitivity
using Random, LinearAlgebra, TRNC, Printf,Roots

function FH_L0Binf()
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


    # also make some noie to fit later
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


    λ = 1.0
    function h_obj(x)
        return norm(x, 0) 
    end


    # put in your initial guesses
    xi = ones(size(pars_FH))
    # this is for l0 norm 
    function prox(q, σ, xk, Δ)

        ProjB(y) = min.(max.(y, xk .- Δ), xk .+ Δ) # define outside? 
        # @show σ/λ, λ
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


    ϵ = 1e-6
    # set all options
    Doptions = s_options(1.0; λ=λ, optTol=ϵ * (1e-6), verbose=0)

    params = TRNCstruct(f_obj, h_obj, λ; FO_options=Doptions, ψχprox=prox, χk=(s) -> norm(s, Inf), HessApprox=LSR1Operator)

    options = TRNCoptions(; maxIter=500, verbose=10, ϵD=ϵ, β=1e16)
    # solve our problem 
    function funcF(x)
        fk = Gradprob(x)
        # @show fk
        if fk == Inf 
            grad = Inf * ones(size(x))
        else
            grad = Zygote.gradient(Gradprob, x)[1] 
        end

        return fk, grad
    end



    # xtr, k, Fhist, Hhist, Comp_pg = TR(xi, params, options)

    function proxl0s(q, σ, xk, Δ)
        # @show σ/λ, λ
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
        s = st - xk
        return s 
    end

    parametersLM = TRNCstruct(f_obj, h_obj, λ; FO_options=Doptions, ψχprox=proxl0s, χk=(s) -> norm(s, Inf))
    optionsLM = TRNCoptions(; σk=1e4, ϵD=ϵ, verbose=10) # options, such as printing (same as above), tolerance, γ, σ, τ, w/e

    # input initial guess
    xlm, klm, Fhistlm, Hhistlm, Comp_pglm = LM(xi, parametersLM, optionsLM)


    @show xtr
    @show xlm
    @show x0



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
end