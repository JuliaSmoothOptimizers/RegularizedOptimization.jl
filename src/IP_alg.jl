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
    Rkprox # ψ_k + χ_k where χ_k is the Δ - norm ball that you project onto. Note that the basic case is that ψ_k = 0
    ψk #nonsmooth model of h that you are trying to solve - it is possible that ψ=h. Otherwise,
                #it's the prox_{ξ*λ*ψ}(s - ν*∇q(s))
    f_obj #objective function (unaltered) that you want to minimize
    h_obj #objective function that is nonsmooth - > only used for evaluation
end

function IP_options(
    ;
    ϵD = 1e-2,
    ϵC = 1e-2,
    Δk = 1.0,
    ptf = 100,
    maxIter = 10000,
    η1 = 1.0e-3, #ρ lower bound
    η2 = 0.9,  #ρ upper bound
    τ = 0.01, #linesearch buffer parameter
    σ = 1.0e-3, # quadratic model linesearch buffer parameter
    γ = 3.0, #trust region buffer
) #default values for trust region parameters in algorithm 4.2
    return IP_params(ϵD, ϵC, Δk, ptf, maxIter,η1, η2, τ, σ, γ)
end

function IP_struct(
    f_obj,
    h;
    FO_options = spg_options(),
    s_alg = minConf_SPG,
    Rkprox = oneProjector,
    ψk = h
)
    return IP_methods(FO_options, s_alg, Rkprox, ψk, f_obj, h)
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
    -Rkprox, function projecting onto the trust region ball or ψ+χ
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
    ϵD = options.ϵD
    ϵC = options.ϵC
    Δk = options.Δk
    ptf = options.ptf
    maxIter = options.maxIter
    η1 = options.η1
    η2 = options.η2 
    σ = options.σ 
    γ = options.γ
    τ = options.τ


    #other parameters
    FO_options = params.FO_options
    s_alg = params.s_alg
    Rkprox = params.Rkprox
    ψk = params.ψk
    f_obj = params.f_obj
    h_obj = params.h_obj


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
        ObjOuter(x) = f_obj(x)[1] + h_obj(x) - μ*sum(log.((x-l).*(u-x)))# - μ * sum(log.(x - l)) - μ * sum(log.(u - x)) #


        k_i = 0
        ρk = -1
        α = 1.0

        #main algorithm initialization
        Fsmth_out = f_obj(xk)
        #test number of outputs to see if user provided a hessian
        (fk, ∇fk, H) = hessdeter(Fsmth_out, xk, k_i)

        # initialize qk
        qk = fk - μ * sum(log.(xk - l)) - μ * sum(log.(u - xk))
        ∇qk = ∇fk - μ ./ (xk - l) + μ ./ (u - xk)
        
        #keep track of old subgradient for LnSrch purposes
        Gν =  ∇fk
        ∇qksj = copy(∇qk) 
        g_old = ((Gν - ∇qksj) + ∇qk) #this is just ∇fk at first 
        #matvec multiplies for hessian 
        ∇²qk = hessmatvec(H, xk, zkl,zku, l, u)
        β = power_iteration(∇²qk,randn(size(xk)))[1]

        kktInit = [norm(g_old - zkl + zku), norm(zkl .* (xk - l) .- μ), norm(zku .* (u - xk) .- μ)]
        kktNorm = 100*kktInit

        # while (kktNorm[1]/kktInit[1] > ϵD || kktNorm[2]/kktInit[2] > ϵC || kktNorm[3]/kktInit[3] > ϵC) && k_i < maxIter
        while (kktNorm[1] > ϵD || kktNorm[2] > ϵC || kktNorm[3] > ϵC) && k_i < maxIter
            #update count
            k_i = k_i + 1 #inner
            k = k + 1  #outer
            TR_stat = ""
            x_stat = ""
            Fobj_hist[k] = fk
            Hobj_hist[k] = h_obj(xk)

            #store previous iterates
            xk⁻ = xk 
            ∇fk⁻ = ∇fk

            


            #define inner function 
            objInner(d) = [0.5*(d'*∇²qk(d)) + ∇qk'*d + qk, ∇²qk(d) + ∇qk] #(mkB, ∇mkB)
            s⁻ = zeros(size(xk))
            
            if typeof(FO_options)!=typeof(spg_options())
                FO_options.β = β
                if h_obj(xk)==0 #i think this is for h==0? 
                    FO_options.λ = Δk * FO_options.β
                end

                (s, s⁻, fsave, funEvals) = s_alg(objInner, s⁻, (d, λν)->Rkprox(d, λν, xk, Δk), FO_options)

                # Gν = (s⁻ - s) * β
                # @show norm((s⁻ - s))
                # Gν = (- s) * β
            else 
                #projects onto ball of radius Δk, weights of 1.0
                (s, fsave, funEvals) = s_alg(objInner, s⁻, (d)->Rkprox(d, 1.0, Δk), FO_options)
                # Gν = (s⁻ - s)/ν = 1/(1/β)(-s) = -(s)β
                # Gν = -s * β   #this isn't quite right for spg_minconf since you technically need the previous g output
            end

            #compute qksj for the previous iterate 
            ∇qksj = ∇qk + ∇²qk(s⁻)



            α = 1.0
            mult = 0.9
            # gradient for z
            dzl = μ ./ (xk - l) - zkl - zkl .* s ./ (xk - l)
            dzu = μ ./ (u - xk) - zku + zku .* s ./ (u - xk)
            # linesearch for step size?
            # if μ!=0
                # α = directsearch(xk - l, u - xk, zkl, zku, s, dzl, dzu)
                α = ls(xk, s,l,u; mult=mult, tau =τ)
                # α = linesearch(xk, zkl, zku, s, dzl, dzu,l,u ;mult=mult, tau = τ)
            # end
            #update search direction for
            s = s * α
            dzl = dzl * α
            dzu = dzu * α

            #define model and update ρ
            mk(d) = 0.5*(d'*∇²qk(d)) + ∇qk'*d + qk + ψk(xk + d) #needs to be xk in the model -> ask user to specify that? 
            # look up how to test if two functions are equivalent? 
            ρk = (ObjOuter(xk) - ObjOuter(xk + s) + 1e-4) / (mk(zeros(size(xk))) - mk(s) + 1e-4)

            if (ρk > η2)
                TR_stat = "increase"
                Δk = max(Δk, γ * norm(s, 1)) #for safety
            else
                TR_stat = "kept"
            end

            if (ρk >= η1 && !(ρk==Inf || isnan(ρk)))
                x_stat = "update"
                xk = xk + s
                zkl = zkl + dzl
                zku = zku + dzu
            end

            if (ρk < η1 || (ρk ==Inf || isnan(ρk)))

                x_stat = "shrink"

                #changed back linesearch
                α = 1.0
                #this needs to be the previous search direction
                while(ObjOuter(xk + α*s) > ObjOuter(xk) + σ*α*(g_old'*s) && α>1e-16) #compute a directional derivative of ψ CHECK LINESEARCH
                    α = α*mult
                    @show α
                end
                # α = 0.1 #was 0.1; can be whatever
                #step should be rejected
                xk = xk + α*s
                zkl = zkl + α*dzl
                zku = zku + α*dzu
                Δk = α * norm(s, 1)
            end

            Fsmth_out = f_obj(xk)

            (fk, ∇fk, H) = hessdeter(Fsmth_out, xk, k_i, s, ∇fk⁻)

            #update qk with new direction
            qk = fk - μ * sum(log.(xk - l)) - μ * sum(log.(u - xk))
            ∇qk = ∇fk - μ ./ (xk - l) + μ ./ (u - xk)
            # ∇²qk(d) = H(d) + Diagonal(zkl ./ (xk - l))*d + Diagonal(zku ./ (u - xk))*d
            ∇²qk = hessmatvec(H, xk, zkl,zku, l, u)


            #update Gν with new direction
            β = power_iteration(∇²qk,randn(size(xk)))[1]
            Gν = (s⁻ - s) * β #is affine scaling of s (αs) still in the subgradient? 
            g_old = (Gν - ∇qksj) + ∇qk
            kktNorm = [
                norm(g_old - zkl + zku) #check this
                norm(zkl .* (xk - l) .- μ)
                norm(zku .* (u - xk) .- μ)
            ]
    
            #Print values
            k % ptf == 0 && 
            @printf(
                "%11d|  %10.5e  %19.5e   %18.5e   %17.5e   %10.5e   %10s   %10.5e   %10s   %10.5e   %10.5e   %10.5e   %10.5e   %10.5e   %10.5e \n",
                # k, μ, kktNorm[1]/kktInit[1],  kktNorm[2]/kktInit[2],  kktNorm[3]/kktInit[3], ρk, x_stat, Δk, TR_stat, α, norm(xk, 2), norm(s, 2), β, fk, ψk(xk))
                k, μ, kktNorm[1],  kktNorm[2],  kktNorm[3], ρk, x_stat, Δk, TR_stat, α, norm(xk, 2), norm(s, 2), β, fk, ψk(xk))

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


function hessdeter(fsmth_output, x, kk, sdes=zeros(size(x)), gradf_prev=zeros(size(x)))
    if length(fsmth_output)==3 #get regular number of outputs
        (f, ∇f, Hess) = fsmth_output
    elseif length(fsmth_output)==2 && kk==0
        (f, ∇f) = fsmth_output #if 2 outputs and if minconf_spg is in play; simple and β should stay in scope 
        if simple ==1
            Hess = I(size(x, 1))
        else
            Hess = β*I(size(x,1))
        end
    elseif length(fsmth_output)==2 && kk>0 #if 2 outputs and you're past the first iterate 
        (f, ∇f) = fsmth_output
        Hess = bfgs_update(Hess, sdes, ∇f-gradf_prev) #update with previous iterate 
    else
        throw(ArgumentError(f_obj, "Function must provide at least 2 outputs - fk and ∇fk. Can also provide Hessian.  ")) #throw error if 1 output or something 
    end

    if isempty(methods(Hess))
        H(d) = Hess*d
    else 
        H = Hess
    end

    return f, ∇f, H
end

function hessmatvec(Hess,x, zl,zu,lb,ub) #l and u should remain in scope here 
    Hessian(d) = Hess(d) + Diagonal(zl ./ (x - lb))*d + Diagonal(zu ./ (ub - x))*d
    return Hessian 
end