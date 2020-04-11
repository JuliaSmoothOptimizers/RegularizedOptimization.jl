#Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".
#Note that some of the file inclusions are for testing purposes (ie minconf_spg)

include("minconf_spg/SPGSlim.jl")
include("minconf_spg/oneProjector.jl")
export IP_options, IntPt_TR, IP_struct #export necessary values to file that calls these functions


mutable struct IP_params
    ϵ #termination criteria
    Δk #trust region radius
    ptf #print every so often
    simple #if you can use spg_minconf with simple projection
    maxIter #maximum amount of inner iterations
end

mutable struct IP_methods
    FO_options #options for minConf_SPG/minimization routine you use for s
    s_alg #algorithm passed that determines descent direction
    χ_projector # Δ - norm ball that you project onto
    prox_ψk
    ψk #part of ϕk that you are trying to solve - for ψ=0, this is just qk. Otherwise,
                #it's the prox_{ξ*λ*ψ}(s - ν*∇q(s))
    f_obj #objective function (unaltered) that you want to minimize
end

function IP_options(;ϵ = 1e-4, Δk=1.0,  ptf = 100, simple=1, maxIter=10000
                      ) #default values for trust region parameters in algorithm 4.2
    return IP_params(ϵ, Δk, ptf, simple, maxIter)
end

function IP_struct(f_obj, h;
    FO_options = spg_options(),s_alg = minConf_SPG, χ_projector=oneProjector,
    ψk=h, prox_ψk=h #prox_{h + δᵦ(x)} for $B = Indicator of \|s\|_p ≦Δ
    )
    return IP_methods(FO_options, s_alg, χ_projector, prox_ψk, ψk, f_obj)
end



"""Interior method for Trust Region problem
    IntPt_TR(x, f_obj, options)
Arguments
----------
x : Array{Float64,1}
    Initial guess for the x value used in the trust region
TotalCount: Float64
    overall count on total iterations
options : mutable structure IP_params with:
    -Δk Float64, trust region radius
    -options, options for descent direction method
    -ptf Int, print output
    -maxIter Float64, maximum number of inner iterations (note: does not influence TotalCount)
Returns
-------
x   : Array{Float64,1}
    Final value of Algorithm 4.2 trust region
k   : Int
    number of iterations used
"""
function IntPt_TR(x0, TotalCount, params, options)

    #initialize passed options
    debug = false #turn this on to see debugging information
    ϵ = options.ϵ
    Δk = options.Δk
    ptf = options.ptf
    simple = options.simple
    maxIter=options.maxIter

    #other parameters
    FO_options = params.FO_options
    s_alg = params.s_alg
    χ_projector = params.χ_projector
    ψk = params.ψk
    prox_ψk = params.prox_ψk
    f_obj = params.f_obj


    #internal variabes
    eta1 = 1.0e-3 #ρ lower bound
    eta2 = 0.9  #ρ upper bound
    tau = 0.01 #linesearch buffer parameter
    sigma = 1.0e-3 # quadratic model linesearch buffer parameter
    gamma = 3.0 #trust region buffer

    #initialize parameters
    xk = copy(x0)

    #make sure you only take the first output of the objective value of the true function you are minimizing
    β(x) = f_obj(x)[1] + ψk(x)#- mu*sum(log.((x-l).*(u-x)))
    #change this to h not psik

    #main algorithm initialization
    (fk, ∇fk, Bk) = f_obj(xk)

    #stopping condition
    Gν = Inf*∇fk
    ∇qk = ∇fk + Bk*zeros(size(∇fk))
    # s = ones(size(∇fk)) #just initialize s
    #norm((g_k + gh_k))
    #g_k∈∂h(xk) -> 1/ν(s_k - s_k^+) // subgradient of your moreau envelope/prox gradient

    if TotalCount==0 #actual first mu
        @printf("---------------------------------------------------------------------------------------------------------------------------------------------------\n")
        @printf("%10s | %11s | %11s | %11s | %10s | %11s | %11s | %10s | %10s | %10s | %10s\n","Iter","Norm((Gν-∇f) + ∇f⁺)","Ratio: ρk", "x status ","TR: Δk", "Δk status", "LnSrch: α", "||x||", "||s||", "f(x)", "h(x)")
        @printf("---------------------------------------------------------------------------------------------------------------------------------------------------\n")
    end

    k_i = 0
    k = TotalCount
    ρk = -1
    α = 1.0

    while(norm((Gν - ∇qk)+ ∇fk) > ϵ && k_i<maxIter)
        #update count
        k_i = k_i+1 #inner
        k = k+1  #outer
        TR_stat = ""
        x_stat = ""

        #define custom inner objective to find search direction and solve

        if simple==1 #when h==0
            objInner(s) = qk(s,fk, ∇fk,Bk) #this can probably be sped up since we declare new function every time
            funProj(x) = χ_projector(x, 1.0, Δk) #projects onto ball of radius Δk, weights of 1.0
            (s, fsave, funEvals)= s_alg(objInner, zeros(size(xk)), funProj, FO_options)
            s⁻ = zeros(size(s))
        else
            FO_options.β = norm(Bk)^2
            FO_options.Bk = Bk
            FO_options.∇fk = ∇fk
            FO_options.xk = xk
            FO_options.Δ = Δk
            funProj = χ_projector
            objInner= prox_ψk
            (s, s⁻, fsave, funEvals)= s_alg(objInner, zeros(size(xk)), funProj, FO_options)

        end
        Gν =(s⁻ - s)/FO_options.β
        ∇qk = ∇fk + Bk*s⁻


        #update ρ
        ########YOU WILL HAVE TO CHANGE THE MODEL TO THE NEW ONE IN THE PAPER###################
        mk(d) = qk(d,fk, ∇fk, Bk)[1] + ψk(xk+d) #qk should take barrier terms into account
        # ρk = (β(xk + s) - β(xk))/(qk(s, ∇Phi,∇²Phi)[1])
        ρk = (β(xk) - β(xk + s))/(mk(zeros(size(xk))) - mk(s)) #test this to make sure it's right (a little variable relative to matlab code)

        if(ρk > eta2)
            TR_stat = "increase"
            Δk = max(Δk, gamma*norm(s, 1)) #for safety
        else
            TR_stat = "kept"
        end

        if(ρk >= eta1)
            x_stat = "update"
            xk = xk + s
        end

        if(ρk < eta1)

            x_stat = "shrink"

            #changed back linesearch
            # α = 1.0
            # while(β(xk + α*s) > β(xk) + sigma*α*∇Phi'*s) #compute a directional derivative of ψ
            #     α = α*mult
            # end
            α = 0.1 #was 0.1; can be whatever
            #step should be rejected
            # xk = xk + α*s
            Δk = α*norm(s, 1)
        end
        # k % ptf ==0 && @printf("%10.5e   %10.5e %10.5e %10.5e\n", β(xk), β(xk + s), mk(zeros(size(xk))), mk(s))

        (fk, ∇fk, Bk) = f_obj(xk);
        #Print values
        k % ptf ==0 && @printf("%11d|  %10.5e   %10.5e   %10s   %10.5e   %10s   %10.5e  %10.5e   %10.5e   %10.5e   %10.5e \n", k, norm((Gν - ∇qk)+ ∇fk), ρk,x_stat, Δk,TR_stat, α, norm(xk,2), norm(s,2), fk, ψk(xk))

        if k % 50 ==0
            FO_options.optTol = FO_options.optTol*.1
        end
        #uncommented for now
        # if(isnan(ρk) || Δk<1e-10)
        #     break
        # end

    end
    return xk, k
end
