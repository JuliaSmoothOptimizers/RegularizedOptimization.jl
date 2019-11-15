#Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization".
#Note that some of the file inclusions are for testing purposes (ie minconf_spg)

include("minconf_spg/SPGSlim.jl")
include("minconf_spg/oneProjector.jl")
# include("Qcustom.jl") #make sure this is here, defines quadratic model for some function; must yield function value, gradient, and hessian
# include("DescentMethods.jl")
export IP_options, IntPt_TR, IP_struct #export necessary values to file that calls these functions


mutable struct IP_params
    epsD #ε bound for 13a, alg 4.3
    epsC #ε bound for 13b, alg 4.2
    trrad #trust region radius
    ptf #print every so often
    simple #if you can use spg_minconf with simple projection
end

mutable struct IP_methods
    l #lower bound
    u #upper bound
    FO_options #options for minConf_SPG/minimization routine you use for s
    s_alg #algorithm passed that determines descent direction
    χ_projector # Δ - norm ball that you project onto
    ϕk #part of ϕk that you are trying to solve - for ψ=0, this is just qk. Otherwise,
                #it's the prox_{ξ*λ*ψ}(s - ν*∇q(s))
    objfun #objective function (unaltered) that you want to minimize
end

function IP_options(;
                      epsD=1.0e-3,
                     epsC = 1.0e-3, trrad=1.0,  ptf = 100, simple=1
                      ) #default values for trust region parameters in algorithm 4.2
    return IP_params(epsD, epsC, trrad, ptf, simple)
end

function IP_struct(objfun; l=Vector{Float64}, u=Vector{Float64},
    FO_options = spg_options(),s_alg = minConf_SPG, χ_projector=oneProjector,
    ϕk = qk
    )
    return IP_methods(l, u, FO_options, s_alg, χ_projector, ϕk, objfun)
end




function IntPt_TR(x, zl, zu,mu,params, options)
    """Return the gradient of the variational penalty objective functional
        IntPt_TR(x, zl, zu,objfun, options)
    Arguments
    ----------
    x : Array{Float64,1}
        Initial guess for the x value used in the trust region
    zl : Array{Float64,1}
        Initial guess for the lower dual parameters
    zu : Array{Float64,1}
        Initial guess for the upper dual parameters
    objfun : generic function with 3 outputs: f(x), gradient(x), Hessian(x)
    options : mutable structure IP_params with:
        -l Array{Float64,1}, lower bound
        -u Array{Float64,1}, upper bound
        -epsD Float64, bound for 13a
        -epsC Float64, bound for 13b
        -trrad Float64, trust region radius
        -options, options for trust region method
        -ptf Int, print output


    Returns
    -------
    x   : Array{Float64,1}
        Final value of Algorithm 4.2 trust region
    zjl : Array{Float64,1}
        final value for the lower dual parameters
    zju : Array{Float64,1}
        final value for the upper dual parameters
    j   : Int
        number of iterations used
    """

    #note - objfun is just l2 norm for the first example, takes in nothing except x. Will generalize later

    #initialize passed options
    debug = false #turn this on to see debugging information
    epsD = options.epsD
    epsC = options.epsC
    trrad = options.trrad
    ptf = options.ptf
    simple = options.simple

    #other parameters
    l = params.l
    u = params.u
    FO_options = params.FO_options
    s_alg = params.s_alg
    χ_projector = params.χ_projector
    ϕk = params.ϕk
    objfun = params.objfun


    #internal variabes
    eta1 = 1.0e-3 #ρ lower bound
    eta2 = 0.9  #ρ upper bound
    tau = 0.01 #linesearch buffer parameter
    sigma = 1.0e-3 # quadratic model linesearch buffer parameter
    gamma = 3.0 #trust region buffer
    zjl = copy(zl)
    zju = copy(zu)


    #make sure you only take the first output of the objective value of the true function you are minimizing
    meritFun(x) = objfun(x)[1] - mu*sum(log.((x-l).*(u-x))) #mu*sum(log.(x-l)) - mu*sum(log.(u-x))

    #main algorithm initialization
    (fj, gj, Hj) = objfun(x)
    kktNorm = [norm(gj - zjl + zju);norm(zjl.*(x-l) .- mu); norm(zju.*(u-x).-mu) ]

    j = 0
    ρj = -1
    α = 1
    while(kktNorm[1] > epsD || kktNorm[2] >epsC || kktNorm[3]>epsC)
        #update count
        j = j+1
        TR_stat = ""
        x_stat = ""

        #compute hessian and gradient for the problem
        ∇Phi = gj - mu./(x-l) + mu./(u-x);
        ∇²Phi = Hj + Diagonal(zjl./(x-l)) + Diagonal(zju./(u-x));


        #define custom inner objective to find search direction and solve

        if simple==1
            objInner(s) = ϕk(s, ∇Phi,∇²Phi ) #this can probably be sped up since we declare new function every time
            funProj(x) = χ_projector(x, 1.0, trrad) #projects onto ball of radius trrad, weights of 1.0
        else
            FO_options.Bk = ∇²Phi
            FO_options.gk = ∇Phi
            FO_options.xk = x
            FO_options.σ_TR = trrad
            objInner(u, ν)= ϕk(u, ν)
            funProj(s, Δ)= χ_projector(s, Δ)
        end
        # funProj(s) = projector(s, trrad, tr_options.β^(-1))

        (s, fsave, funEvals)= s_alg(objInner, zeros(size(x)), funProj, FO_options)


        # gradient for z
        dzl = mu./(x-l) - zjl - zjl.*s./(x-l)
        dzu = mu./(u-x) - zju + zju.*s./(u-x)

        α = 1.0
        mult = 0.9

        #linesearch to adjust parameter
        # α = linesearch(x, zjl, zju, s, dzl, dzu; mult=mult, tau = tau)
        # α = directsearch(x, zjl, zju, s, dzl, dzu)
        directsearch!(x-l, u-x, α,zjl, zju, s, dzl, dzu)

        #update search direction for
        s = s*α
        dzl = dzl*α
        dzu = dzu*α

        #update ρ
        ρj = (meritFun(x + s) - meritFun(x))/(objInner(s)[1]) #test this to make sure it's right (a little variable relative to matlab code)

        if(ρj > eta2)
            TR_stat = "increase"
            trrad = max(trrad, gamma*norm(s, 1)) #for safety
        else
            TR_stat = "kept"
        end

        if(ρj >= eta1)
            x_stat = "update"
            x = x + s
            zjl = zjl + dzl
            zju = zju + dzu
        end

        if(ρj < eta1)

            x_stat = "shrink"

            α = 1.0;
            while(meritFun(x + α*s) > meritFun(x) + sigma*α*∇Phi'*s)
                α = α*mult;
            end
            x = x + α*s;
            zjl = zjl + α*dzl;
            zju = zjl + α*dzu;
            trrad = α*norm(s, 1);
        end


        (fj, gj, Hj) = objfun(x);
        kktNorm = [norm(gj - zjl + zju);norm(zjl.*(x-l) .- mu); norm(zju.*(u-x).-mu) ]
                #Print values
        j % ptf ==0 && @printf("Iter %4d, Norm(kkt) %1.5e, ρj %1.5e/%s, trustR %1.5e/%s, mu %1.5e, α %1.5e\n", j, sum(kktNorm), ρj,x_stat, trrad,TR_stat, mu, α)


    end
    return x, zjl, zju, j
end
