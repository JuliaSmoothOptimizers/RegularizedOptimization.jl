#Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".
#Note that some of the file inclusions are for testing purposes (ie minconf_spg)

include("minconf_spg/SPGSlim.jl")
include("minconf_spg/oneProjector.jl")
export IP_options, IntPt_TR, IP_struct #export necessary values to file that calls these functions


mutable struct IP_params
    epsD #ε bound for 13a, alg 4.3
    epsC #ε bound for 13b, alg 4.2
    Δk #trust region radius
    ptf #print every so often
    simple #if you can use spg_minconf with simple projection
end

mutable struct IP_methods
    l #lower bound
    u #upper bound
    FO_options #options for minConf_SPG/minimization routine you use for s
    s_alg #algorithm passed that determines descent direction
    χ_projector # Δ - norm ball that you project onto
    prox_ψk
    ψk #part of ϕk that you are trying to solve - for ψ=0, this is just qk. Otherwise,
                #it's the prox_{ξ*λ*ψ}(s - ν*∇q(s))
    f_obj #objective function (unaltered) that you want to minimize
end

function IP_options(;
                      epsD=1.0e-3,
                     epsC = 1.0e-3, Δk=1.0,  ptf = 100, simple=1
                      ) #default values for trust region parameters in algorithm 4.2
    return IP_params(epsD, epsC, Δk, ptf, simple)
end

function IP_struct(f_obj, h; l=Vector{Float64}, u=Vector{Float64},
    FO_options = spg_options(),s_alg = minConf_SPG, χ_projector=oneProjector,
    ψk=h, prox_ψk=h #initialize
    )
    return IP_methods(l, u, FO_options, s_alg, χ_projector, prox_ψk, ψk, f_obj)
end




function IntPt_TR(x0, zl0, zu0,mu, TotalCount, params, options)
    """Return the gradient of the variational penalty objective functional
        IntPt_TR(x, zl, zu,f_obj, options)
    Arguments
    ----------
    x : Array{Float64,1}
        Initial guess for the x value used in the trust region
    zl : Array{Float64,1}
        Initial guess for the lower dual parameters
    zu : Array{Float64,1}
        Initial guess for the upper dual parameters
    f_obj : generic function with 3 outputs: f(x), gradient(x), Hessian(x)
    options : mutable structure IP_params with:
        -l Array{Float64,1}, lower bound
        -u Array{Float64,1}, upper bound
        -epsD Float64, bound for 13a
        -epsC Float64, bound for 13b
        -Δk Float64, trust region radius
        -options, options for trust region method
        -ptf Int, print output
    Returns
    -------
    x   : Array{Float64,1}
        Final value of Algorithm 4.2 trust region
    zkl : Array{Float64,1}
        final value for the lower dual parameters
    zku : Array{Float64,1}
        final value for the upper dual parameters
    k   : Int
        number of iterations used
    """

    #initialize passed options
    debug = false #turn this on to see debugging information
    epsD = options.epsD
    epsC = options.epsC
    Δk = options.Δk
    ptf = options.ptf
    simple = options.simple

    #other parameters
    l = params.l
    u = params.u
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
    zkl = copy(zl0)
    zku = copy(zu0)
    xk = copy(x0)

    #make sure you only take the first output of the objective value of the true function you are minimizing
    meritFun(x) = f_obj(x)[1] - mu*sum(log.((x-l).*(u-x))) + ψk(x) #mu*sum(log.(x-l)) - mu*sum(log.(u-x))

    #main algorithm initialization
    (fk, gk, Hk) = f_obj(xk)
    kktNorm = [norm(gk - zkl + zku); norm(zkl.*(xk-l) .- mu); norm(zku.*(u-xk).-mu)]


    if mu == 1
        @printf("----------------------------------------------------------------------------------------------------------------\n")
        @printf("%10s | %11s | %11s | %11s | %11s | %11s | %11s | %11s %11s %11s %11s\n","Iter","Norm(kkt)","Ratio: ρk", "x status ","TR: Δk", "Δk status","Barrier: μk", "LnSrch: α", "||x||", "||s||", "f(x)+h(x)")
        @printf("----------------------------------------------------------------------------------------------------------------\n")
    end

    k = TotalCount
    ρk = -1
    α = 1
    while(kktNorm[1] > epsD || kktNorm[2] >epsC || kktNorm[3]>epsC)
        #update count
        k = k+1
        TR_stat = ""
        x_stat = ""

        #compute hessian and gradient for the problem
        ∇Phi = gk - mu./(xk-l) + mu./(u-xk)
        ∇²Phi = Hk + Diagonal(zkl./(xk-l)) + Diagonal(zku./(u-xk))


        #define custom inner objective to find search direction and solve

        if simple==1
            objInner(s) = qk(s, ∇Phi,∇²Phi ) #this can probably be sped up since we declare new function every time
            funProj(x) = χ_projector(x, 1.0, Δk) #projects onto ball of radius Δk, weights of 1.0
        else
            FO_options.Bk = ∇²Phi
            FO_options.gk = ∇Phi
            FO_options.xk = xk
            FO_options.σ_TR = Δk
            funProj = χ_projector
            objInner= prox_ψk
        end
        (s, fsave, funEvals)= s_alg(objInner, zeros(size(xk)), funProj, FO_options)


        # gradient for z
        dzl = mu./(xk-l) - zkl - zkl.*s./(xk-l)
        dzu = mu./(u-xk) - zku + zku.*s./(u-xk)

        α = 1.0
        mult = 0.9

        #linesearch to adjust parameter
        # α = linesearch(xk, zkl, zku, s, dzl, dzu,l,u; mult=mult, tau = tau)
        α = directsearch(xk-l, u-xk ,zkl, zku, s, dzl, dzu)
        # directsearch!(xk-l, u-xk, α,zkl, zku, s, dzl, dzu) #alpha to the boundary

        #update search direction for
        s = s*α
        dzl = dzl*α
        dzu = dzu*α

        #update ρ
        mk(d) = qk(d, ∇Phi, ∇²Phi)[1] + ψk(xk+d) #qk should take barrier into account
        # ρk = (meritFun(xk + s) - meritFun(xk))/(qk(s, ∇Phi,∇²Phi)[1])
        ρk = (meritFun(xk) - meritFun(xk + s))/(mk(zeros(size(xk))) - mk(s)) #test this to make sure it's right (a little variable relative to matlab code)

        if(ρk > eta2)
            TR_stat = "increase"
            Δk = max(Δk, gamma*norm(s, 1)) #for safety
        else
            TR_stat = "kept"
        end

        if(ρk >= eta1)
            x_stat = "update"
            xk = xk + s
            zkl = zkl + dzl
            zku = zku + dzu
        end

        if(ρk < eta1) #

            x_stat = "shrink"

            #changed back linesearch
            # α = 1.0
            # while(meritFun(xk + α*s) > meritFun(xk) + sigma*α*∇Phi'*s) #compute a directional derivative of ψ
            #     α = α*mult
            # end
            α = 0.1
            xk = xk + α*s
            zkl = zkl + α*dzl
            zku = zku + α*dzu
            Δk = α*norm(s, 1)
        end


        (fk, gk, Hk) = f_obj(xk);
        kktNorm = [norm(gk - zkl + zku);norm(zkl.*(xk-l) .- mu); norm(zku.*(u-xk).-mu) ]
        #Print values
        k % ptf ==0 && @printf("%11d|  %10.5e   %10.5e   %10s   %10.5e   %10s   %10.5e   %10.5e| %10.5e %10.5e %10.5e \n", k, sum(kktNorm), ρk,x_stat, Δk,TR_stat, mu, α, norm(xk,2), norm(s,2), fk+ψk(xk))


    end
    return xk, zkl, zku, k
end
