#Implements Algorithm 4.2 in "Interior-Point Trust-Region Method for Composite Optimization". 
#Note that some of the file inclusions are for testing purposes (ie minconf_spg)

using LinearAlgebra, Printf #include necessary packages
include("minconf_spg/oneProjector.jl")
include("Qcustom.jl") #make sure this is here, defines quadratic model for some function; must yield function value, gradient, and hessian
export IP_options, IntPt_TR #export necessary values to file that calls these functions


mutable struct IP_params
    l #lower bound 
    u #upper bound
    mu #barrier parameter 
    epsD #ε bound for 13a, alg 4.3
    epsC #ε bound for 13b, alg 4.2
    trrad #trust region radius
    minconf_options #options for minConf_SPG, can change later 
    ptf #print every so often 
end

function IP_options(;
                     l=Array{Float64,1}(undef,0), u=Array{Float64,1}(undef,0), mu = 1e-3, epsD=1e-2,
                     epsC = 1e-2, trrad=1.0, minconf_options = spg_options(), ptf = 100
                      ) #default values for trust region parameters in algorithm 4.2
    return IP_params(l, u, mu, epsD, epsC, trrad, minconf_options, ptf)
end

function IntPt_TR(x, zl, zu,objfun, options)
    r"""Return the gradient of the variational penalty objective functional

    Parameters
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
        -minconf_options, options for minConf_SPG
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
    debug = false; #turn this on to see debugging information 
    l = options.l;
    u = options.u;
    mu = options.mu;
    epsD = options.epsD;
    epsC = options.epsC; 
    trrad = options.trrad;
    minconf_options = options.minconf_options; 
    ptf = options.ptf; 


    #internal variabes
    eta1 = 1.0e-4 #ρ lower bound
    eta2 = 0.9  #ρ upper bound 
    tau = 0.01 #linesearch buffer parameter 
    sigma = 1.0e-4 # quadratic model linesearch buffer parameter 
    gamma = 3.0 #trust region buffer 
    zjl = copy(zl)
    zju = copy(zu)


    #make sure you only take the first output of the objective value of the true function you are minimizing
    meritFun(x) = objfun(x)[1] - mu*sum(log.(x-l)) - mu*sum(log.(u-x));


    #main algorithm initialization 
    (fj, gj, Hj) = objfun(x)
    j = 0
    ρj = -1
    α = 1
    while(norm(gj - zjl + zju) > epsD || norm(zjl.*(x-l) .- mu)>epsC || norm(zju.*(u-x).-mu)>epsC)
        #update count
        j = j+1

        #look at kkt norm
        kktNorm = norm(gj - zjl + zju) + norm(zjl.*(x-l) .- mu) + norm(zju.*(u-x) .- mu)

       

        #Print values 
        j % ptf ==0 && @printf("Iter %4d, Norm(kkt) %1.5e, ρj %1.5e, trustR %1.5e, mu %1.5e, α %1.5e\n", j, kktNorm, ρj, trrad, mu, α)



        #compute hessian and gradient for the problem 
        ∇Phi = gj - mu./(x-l) + mu./(u-x);
        ∇²Phi = Hj + Diagonal(zjl./(x-l)) + Diagonal(zju./(u-x));

        par = Q_params(obj=fj, grad =∇Phi, Hess=∇²Phi)
        # par.Hess = ∇²Phi
        # par.grad = ∇Phi

        #define custom inner objective to find search direction and solve
        
        objInner(s) = QCustom(s, par) #this can probably be sped up since we declare new function every time 
        funProj(x) = oneProjector(x, 1.0, trrad)
        
        (s, fsave, funEvals)= minConf_SPG(objInner, zeros(size(x)), funProj, minconf_options)

        
        # gradient for z
        dzl = mu./(x-l) - zjl - zjl.*s./(x-l)
        dzu = mu./(u-x) - zju + zju.*s./(u-x)

        α = 1.0
        mult = 0.9
        
        #linesearch to adjust parameter
        while( 
            any(x + α*s - l .< (1-tau)*(x-l)) || 
            any(u - x - α*s .< (1-tau)*(u-x)) ||
            any(zjl + α*dzl .< (1-tau)*zjl) || 
            any(zju + α*dzu .< (1-tau)*zju)
            )
            α = α*mult
        end
        
        #update search direction for 
        s = s*α
        dzl = dzl*α
        dzu = dzu*α

        #update ρ
        ρj = (meritFun(x + s) - meritFun(x))/(objInner(s)[1]) #test this to make sure it's right (a little variable relative to matlab code)
        if(debug)
            @printf("rhoj is %1.5e\n", ρj)
        end
        
        if(ρj > eta2)
            if(debug)
                @printf("increase\n")
            end
            trrad = max(trrad, gamma*norm(s, 1)) #for safety
        end
        
        if(ρj >= eta1)
            if(debug)
                @printf("update\n")
            end
            x = x + s;
            zjl = zjl + dzl;
            zju = zju + dzu;
        end
        
        if(ρj < eta1)
            if(debug)
                @printf("shrink\n")
            end
            α = 1.0;
            while(meritFun(x + α*s) > meritFun(x) + sigma*α*∇Phi'*s)
                alpha = α*mult;
            end
            x = x + α*s;
            zjl = zjl + α*dzl;
            zju = zjl + α*dzu;
            trrad = α*norm(s, 1);
        end
        
        if(debug)
            @printf("trrad is : %5.2e\n", trrad);
        end
        (fj, gj, Hj) = objfun(x);

    end
    return x, zjl, zju, j
end