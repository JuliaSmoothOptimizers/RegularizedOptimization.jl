#Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization". 
#Note that some of the file inclusions are for testing purposes (ie minconf_spg)

using LinearAlgebra, Printf #include necessary packages
# include("minconf_spg/SPGSlim.jl")
include("DescentMethods.jl")
include("Qcustom.jl") #make sure this is here, defines quadratic model for some function; must yield function value, gradient, and hessian
export IP_options, IntPt_TR, IP_struct #export necessary values to file that calls these functions


mutable struct IP_params
    epsD #ε bound for 13a, alg 4.3
    epsC #ε bound for 13b, alg 4.2
    trrad #trust region radius
    ptf #print every so often 
end

mutable struct IP_methods
    l #lower bound 
    u #upper bound 
    tr_options #options for minConf_SPG
    tr_projector_alg #algorithm passed that determines
    projector # norm ball that you project onto 
    objfun #objective function  
end

function IP_options(;
                      epsD=1.0e-3,
                     epsC = 1.0e-3, trrad=1.0,  ptf = 100
                      ) #default values for trust region parameters in algorithm 4.2
    return IP_params(epsD, epsC, trrad, ptf)
end

function IP_struct(objfun; l=Vector{Float64}, u=Vector{Float64}, tr_options = spg_options(),tr_projector_alg = minConf_SPG,projector=oneProjector
    )
    return IP_methods(l, u, tr_options, tr_projector_alg, projector, objfun)
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

    #other parameters 
    l = params.l
    u = params.u
    tr_options = params.tr_options 
    tr_projector_alg = params.tr_projector_alg
    projector = params.projector 
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

        par = Q_params(grad =∇Phi, Hess=∇²Phi)
        # par.Hess = ∇²Phi
        # par.grad = ∇Phi

        #define custom inner objective to find search direction and solve
        
        objInner(s) = QCustom(s, par) #this can probably be sped up since we declare new function every time 
        funProj(x) = projector(x, 1.0, trrad) #projects onto ball of radius trrad, weights of 1.0
        # funProj(s) = projector(s, trrad, tr_options.β^(-1))
        
        (s, fsave, funEvals)= tr_projector_alg(objInner, zeros(size(x)), funProj, tr_options)

        
        # gradient for z
        dzl = mu./(x-l) - zjl - zjl.*s./(x-l)
        dzu = mu./(u-x) - zju + zju.*s./(u-x)

        α = 1.0
        mult = 0.9
        
        #linesearch to adjust parameter
        # α = linesearch(x, zjl, zju, s, dzl, dzu; mult=mult, tau = tau)
        
        # α = directsearch(x, zjl, zju, s, dzl, dzu) 
        directsearch!(α,zjl, zju, s, dzl, dzu) 
        # @printf("%1.5e | %1.5e \n", α1, α)

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