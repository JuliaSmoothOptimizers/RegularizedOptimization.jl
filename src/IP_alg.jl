#Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".
#Note that some of the file inclusions are for testing purposes (ie minconf_spg)

include("minconf_spg/SPGSlim.jl")
include("minconf_spg/oneProjector.jl")
include("PowerIter.jl")
using Plots
export IP_options, IntPt_TR, IP_struct #export necessary values to file that calls these functions


mutable struct IP_params
    ϵD #termination criteria
    ϵC #dual termination criteria
    Δk #trust region radius
    ptf #print every so often
    simple #if you can use spg_minconf with simple projection
    maxIter #maximum amount of inner iterations
    η1 #ρ lower bound 
    η2 #ρ upper bound 
    τ # linesearch buffer parameter 
    σ #quadratic model linesearch buffer parameter
    γ #trust region buffer 
end

mutable struct IP_methods
    FO_options #options for minConf_SPG/minimization routine you use for s
    s_alg #algorithm passed that determines descent direction
    χ_projector # Δ - norm ball that you project onto
    InnerFunc
    ψk #part of ϕk that you are trying to solve - for ψ=0, this is just qk. Otherwise,
                #it's the prox_{ξ*λ*ψ}(s - ν*∇q(s))
    f_obj #objective function (unaltered) that you want to minimize
end

function IP_options(
    ;
    ϵD = 1e-2,
    ϵC = 1e-2,
    Δk = 1.0,
    ptf = 100,
    simple = 1,
    maxIter = 10000,
    η1 = 1.0e-3, #ρ lower bound
    η2 = 0.9,  #ρ upper bound
    τ = 0.01, #linesearch buffer parameter
    σ = 1.0e-3, # quadratic model linesearch buffer parameter
    γ = 3.0, #trust region buffer
) #default values for trust region parameters in algorithm 4.2
    return IP_params(ϵD, ϵC, Δk, ptf, simple, maxIter,η1, η2, τ, σ, γ)
end

function IP_struct(
    f_obj,
    h;
    FO_options = spg_options(),
    s_alg = minConf_SPG,
    χ_projector = oneProjector,
    ψk = h,
    InnerFunc = h, #prox_{h + δᵦ(x)} for $B = Indicator of \|s\|_p ≦Δ
)
    return IP_methods(FO_options, s_alg, χ_projector, InnerFunc, ψk, f_obj)
end



"""Interior method for Trust Region problem
    IntPt_TR(x, TotalCount,params, options)
Arguments
----------
x : Array{Float64,1}
    Initial guess for the x value used in the trust region
TotalCount: Float64
    overall count on total iterations
params : mutable structure IP_params with:
    --
    -ϵD, tolerance for primal convergence
    -ϵC, tolerance for dual convergence
    -Δk Float64, trust region radius
    -ptf Int, print every # iterations
    -simple, 1 for h=0, 0 for other
    -maxIter Float64, maximum number of inner iterations (note: does not influence TotalCount)
options : mutable struct IP_methods
    -f_obj, smooth objective function; takes in x and outputs [f, g, Bk]
    -h_obj, nonsmooth objective function; takes in x and outputs h
    --
    -FO_options, options for first order algorithm, see DescentMethods.jl for more
    -s_alg, algorithm for descent direction, see DescentMethods.jl for more
    -χ_projector, function projecting onto the trust region ball or h+χ
    -InnerFunc, inner objective or proximal operator of ψk+χk+1/2||u - sⱼ + ∇qk|²
l : Vector{Float64} size of x, defaults to -Inf
u : Vector{Float64} size of x, defaults to Inf
μ : Float64, initial barrier parameter, defaults to 1.0

Returns
-------
x   : Array{Float64,1}
    Final value of Algorithm 4.2 trust region
k   : Int
    number of iterations used
"""
function IntPt_TR(
    x0,
    params,
    options;
    l = -1.0e16 * ones(size(x0)),
    u = 1.0e16 * ones(size(x0)),
    μ = 0.0,
    BarIter = 1,
)

    #initialize passed options
    debug = false #turn this on to see debugging information
    ϵD = options.ϵD
    ϵC = options.ϵC
    Δk = options.Δk
    ptf = options.ptf
    simple = options.simple
    maxIter = options.maxIter
    η1 = options.η1
    η2 = options.η2 
    σ = options.σ 
    γ = options.γ
    τ = options.τ


    #other parameters
    FO_options = params.FO_options
    s_alg = params.s_alg
    χ_projector = params.χ_projector
    ψk = params.ψk
    InnerFunc = params.InnerFunc
    f_obj = params.f_obj


    #initialize parameters
    xk = copy(x0)
    #initialize them to positive values for x=l and negative for x=u
    zkl = ones(size(x0))
    zku = -ones(size(x0))
    k = 0
    Fobj_hist = zeros(maxIter * BarIter)
    Hobj_hist = zeros(maxIter * BarIter)
    @printf(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n",
    )
    @printf(
        "%10s | %11s | %11s | %11s | %11s | %11s | %10s | %11s | %11s | %10s | %10s | %10s | %10s   | %10s | %10s\n",
        "Iter",
        "μ",
        "||(Gν-∇q) + ∇ϕ⁺)-zl+zu||",
        "||zl(x-l) - μ||",
        "||zu(u-x) - μ||",
        "Ratio: ρk",
        "x status ",
        "TR: Δk",
        "Δk status",
        "LnSrch: α",
        "||x||",
        "||s||",
        "β",
        "f(x)",
        "h(x)",
    )
    @printf(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n",
    )

#Barrier Loop
    while k < BarIter && (μ > 1e-6 || μ==0) #create options for this
        #make sure you only take the first output of the objective value of the true function you are minimizing
        β(x) = f_obj(x)[1] + ψk(x) - μ*sum(log.((x-l).*(u-x)))# - μ * sum(log.(x - l)) - μ * sum(log.(u - x)) #
        #change this to h not psik

        #main algorithm initialization
        (fk, ∇fk, Bk) = f_obj(xk)
        ϕ = fk - μ * sum(log.(xk - l)) - μ * sum(log.(u - xk))
        ∇ϕ = ∇fk - μ ./ (xk - l) + μ ./ (u - xk)
        # H = Bk + Diagonal(zkl ./ (xk - l)) + Diagonal(zku ./ (u - xk))
        # ∇²ϕ(s) = Bk(s) + Diagonal(zkl ./ (xk - l))*s + Diagonal(zku ./ (u - xk))*s
        #stopping condition
        Gν =  ∇fk
        ∇qk = ∇ϕ
        # s = ones(size(∇fk)) #just initialize s
        #norm((g_k + gh_k))
        #g_k∈∂h(xk) -> 1/ν(s_k - s_k^+) // subgradient of your moreau envelope/prox gradient

        k_i = 0
        ρk = -1
        α = 1.0
        kktNorm = [norm(((Gν - ∇qk) + ∇ϕ) - zkl + zku), norm(zkl .* (xk - l) .- μ), norm(zku .* (u - xk) .- μ)]
        kktInit = kktNorm
        # @printf("%10.5e   %10.5e %10.5e %10.5e\n",kktNorm[1], kktNorm[2], kktNorm[3],k_i)
        # while(norm((Gν - ∇qk)+ ∇fk) > ϵD && k_i<maxIter)
        while (kktNorm[1]/kktInit[1] > ϵD || kktNorm[2]/kktInit[2] > ϵC || kktNorm[3]/kktInit[3] > ϵC) && k_i < maxIter
            #update count
            k_i = k_i + 1 #inner
            k = k + 1  #outer
            TR_stat = ""
            x_stat = ""
            Fobj_hist[k] = fk
            Hobj_hist[k] = ψk(xk)


            (fk, ∇fk, Bk) = f_obj(xk)
            ϕ = fk - μ * sum(log.(xk - l)) - μ * sum(log.(u - xk))
            ∇ϕ = ∇fk - μ ./ (xk - l) + μ ./ (u - xk)
            # H = Bk + Diagonal(zkl ./ (xk - l)) + Diagonal(zku ./ (u - xk))
            ∇²ϕ(s) = Bk(s) + Diagonal(zkl ./ (xk - l))*s + Diagonal(zku ./ (u - xk))*s


            k % ptf == 0 && @printf(
                "%11d|  %10.5e  %19.5e   %18.5e   %17.5e   %10.5e   %10s   %10.5e   %10s   %10.5e   %10.5e   %10.5e   %10.5e   %10.5e   %10.5e \n",
                k, μ, kktNorm[1]/kktInit[1],  kktNorm[2]/kktInit[2],  kktNorm[3]/kktInit[3], ρk, x_stat, Δk, TR_stat, α, norm(xk, 2), norm(s, 2),power_iteration(∇²ϕ, randn(size(s)))[1], fk, ψk(xk))


            # @printf("%10.5e   %10.5e %10.5e %10.5e\n",norm(∇²ϕ - Bk), norm(∇ϕ - ∇fk), norm(fk - ϕ), norm(∇qk - ∇fk))

            if simple == 1 || simple == 2
                # objInner(s) = qk(s, ϕ, ∇ϕ, ∇²ϕ)[1:2]
                objInner(s) = [0.5*(s'*∇²ϕ(s)) + ∇ϕ'*s + fk, ∇²ϕ(s) + ∇ϕ]

            else
                objInner = InnerFunc
            end

            if simple == 1
                funProj(s) = χ_projector(s, 1.0, Δk) #projects onto ball of radius Δk, weights of 1.0
                s⁻ = zeros(size(xk))
                (s, fsave, funEvals) = s_alg(objInner, s⁻, funProj, FO_options)
                # Gν = -s * eigmax(H) #Gν = (s⁻ - s)/ν = 1/(1/β)(-s) = -(s)β
                Gν = -s * power_iteration(∇²ϕ, randn(size(xk)))[1]
                #this can probably be sped up since we declare new function every time
            else
                FO_options.β = eigmax(H)
                # FO_options.β = power_iteration(∇²ϕ, randn(size(xk)))[1]
                FO_options.Bk = ∇²ϕ
                FO_options.∇fk = ∇ϕ
                FO_options.xk = xk
                FO_options.Δ = Δk
                s⁻ = zeros(size(xk))
                if simple == 2
                    FO_options.λ = Δk * eigmax(H)
                    # FO_options.λ = Δk * power_iteration(∇²ϕ, randn(size(xk)))[1]
                end
                funProj = χ_projector
                (s, s⁻, fsave, funEvals) = s_alg(
                    objInner,
                    s⁻,
                    funProj,
                    FO_options,
                )
                Gν = (s⁻ - s) * FO_options.β
            end

            ∇qk = ∇ϕ + ∇²ϕ(s⁻)


            α = 1.0
            mult = 0.9
            # gradient for z
            dzl = μ ./ (xk - l) - zkl - zkl .* s ./ (xk - l)
            dzu = μ ./ (u - xk) - zku + zku .* s ./ (u - xk)

            # @printf("%10.5e   %10.5e %10.5e  %10.5e %10.5e %10.5e\n",α, norm(s), norm(dzl),norm(dzu), norm(s⁻), norm(Gν))
      
            # linesearch for step size?
            # if μ!=0
                # α = directsearch(xk - l, u - xk, zkl, zku, s, dzl, dzu)
                α = ls(xk, s,l,u ;mult=mult, tau =τ)
                # α = linesearch(xk, zkl, zku, s, dzl, dzu,l,u ;mult=mult, tau = τ)
            # end
            # @printf("%10.5e   %10.5e %10.5e  %10.5e %10.5e %10.5e\n",α, α1, minimum(xk-l), minimum(xk-u), minimum(zkl*τ + dzl), minimum(zku*τ + dzu))
            #update search direction for
            s = s * α
            dzl = dzl * α
            dzu = dzu * α

            #update ρ
            # mk(d) = qk(d, ϕ, ∇ϕ, ∇²ϕ)[1] + ψk(xk + d) #qk should take barrier terms into account
            mk(d) = 0.5*(d'*∇²ϕ(d)) + ∇ϕ'*d + fk
            # ρk = (β(xk + s) - β(xk))/(qk(s, ∇Phi,∇²Phi)[1])
            ρk = (β(xk) - β(xk + s) + 1e-4) / (mk(zeros(size(xk))) - mk(s) + 1e-4)
            # @printf("%10.5e   %10.5e %10.5e %10.5e\n", maximum(s),maximum(xk+s), minimum(s), minimum(xk+s))
            # @printf("%10.5e   %10.5e %10.5e %10.5e\n", norm(s,0), norm(xk+s,0), norm(xk,0),γ * norm(s, 1))
            if (ρk > η2)
                TR_stat = "increase"
                Δk = max(Δk, γ * norm(s, 1)) #for safety
            else
                TR_stat = "kept"
            end

            if (ρk >= η1)
                x_stat = "update"
                xk = xk + s
                zkl = zkl + dzl
                zku = zku + dzu
            end

            if (ρk < η1)

                x_stat = "shrink"

                #changed back linesearch
                # α = 1.0
                # while(β(xk + α*s) > β(xk) + σ*α*(∇ϕ + (Gν - ∇qk))'*s) #compute a directional derivative of ψ
                #     α = α*mult
                # end
                α = 0.1 #was 0.1; can be whatever
                #step should be rejected
                xk = xk + α*s
                zkl = zkl + α*dzl
                zku = zku + α*dzu
                Δk = α * norm(s, 1)
            end

            # @printf("%10.5e   %10.5e %10.5e %10.5e\n",norm(Gν), norm(Gν-∇qk), FO_options.β, norm(s⁻ - s))
            kktNorm = [
                norm(((Gν - ∇qk) + ∇ϕ) - zkl + zku) #check this
                norm(zkl .* (xk - l) .- μ)
                norm(zku .* (u - xk) .- μ)
            ]

            # plot(xk, xlabel="i^th index", ylabel="x", title="x Progression", label="x_k")
            # plot!(xk-s, label="x_(k-1)", marker=2)
            # filestring = string("figs/bpdn/LS_l0_Binf/xcomp", k, ".pdf")
            # savefig(filestring)       
            #Print values

            if k % ptf == 0
                FO_options.optTol = FO_options.optTol * 0.1
            end
        end
        # mu = norm(zl.*(x.-l)) + norm(zu.*(u.-x))
        μ = 0.1 * μ
        k = k + 1
        ϵD = ϵD * μ
        ϵC = ϵC * μ

    end
    return xk, k, Fobj_hist, Hobj_hist
end
