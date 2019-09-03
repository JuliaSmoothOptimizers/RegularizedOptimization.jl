using LinearAlgebra, Random, Printf
# include("minconf_spg/SLIM_optim.jl")
include("minconf_spg/oneProjector.jl")
include("Qcustom.jl")
export IP_options, IntPt_TR


mutable struct IP_params
    m
    n
    A
    b
    l
    u
    mu
    epsD
    epsC
    trrad
    minconf_options
    ptf
end

function IP_options(;m=100, n=100, A=Array{Float64,2}(undef, 0, 0), b =Array{Float64,1}(undef,0),
                     l=Array{Float64,1}(undef,0), u=Array{Float64,1}(undef,0), mu = 1e-3, epsD=1e-2,
                     epsC = 1e-2, trrad=1.0, minconf_options = spg_options(), ptf = 100
                      )
    return IP_params(m,n, A, b, l, u, mu, epsD, epsC, trrad, minconf_options, ptf)
end

function IntPt_TR(x, zl, zu,objfun, options)

    #note - objfun is just l2 norm for the first example, takes in nothing except x. Will generalize later

    #initialize internal variables
    debug = 1;
    l = options.l;
    u = options.u;
    mu = options.mu;
    epsD = options.epsD;
    epsC = options.epsC; 
    trrad = options.trrad;
    minconf_options = options.minconf_options; 
    ptf = options.ptf; 


    #internal variabes
    eta1 = 1.0e-4
    eta2 = 0.9
    tau = 0.01
    sigma = 1.0e-4
    gamma = 3.0
    zjl = copy(zl)
    zju = copy(zu)



    meritFun(x) = objfun(x) - mu*sum(log(x-l)) - mu*sum(log(u-x));


    #main algorithm
    (fj, gj, Hj) = objfun(x)


    j = 0
    ρj = -1
    α = 1
    while(norm(gj - zjl + zju) > epsD || norm(zjl.*(x-l) - mu)>epsC || norm(zju.*(u-x)-mu)>epsC)
        #update count
        j = j+1

        #look at kkt norm
        kktNorm = norm(gj - zjl + zju) + norm(zjl.*(x-l)-mu*ones(size(x))) + norm(zju.*(u-x) - mu*ones(size(x)))

       


        j % ptf ==0 && @printf("Iter %4d, Norm(kkt) %1.5e, ρ_{j} %1.5e, trustR %1.5e, mu %1.5e, α %1.5e\n", j, kktNorm, ρj, trrad, mu, α)



        #compute hessian and gradient for the problem
        ∇Phi = gj - mu./(x-l) + mu./(u-x)
        ∇²Phi = Hj + Diagonal(zjl./(x-l)) + Diagonal(zju./(u-x))

        par = Q_params(grad =∇Phi, Hess=∇²Phi)
        # par.Hess = ∇²Phi
        # par.grad = ∇Phi

        #define custom inner objective to find search direction and solve
        objInner(s) = QCustom(s, par)
        funProj(x) = oneProjector(x, 1, trrad)
        (s, fsave, funEvals)= minConf_SPG(objInner, 0.0*x, funProj, minconf_options)

        # gradient for z
        dzl = mu./(x-l) - zjl - zjl.*s./(x-l)
        dzu = mu./(u-x) - zju + zju.*s./(u-x)

        α = 1.0
        mult = 0.9

        #linesearch to adjust parameter
        while(any(x + α*s - l < (1-tau)*(x-l)) || 
            any(u - x - α*s < (1-tau)*(u-x)) ||
            any(zjl + α*dzl < (1-tau)*zjl) || 
            any(zju + α*dzu < (1-tau)*zju)
            )
            α = α*mult
        end
        #update search direction for 
        s = s*α
        dzl = dxl*α
        dzu = dzu*α

        #update ρ
        ρj = (meritfun(x + s) - meritfun(x)/objInner(s))
        if(debug)
            @printf("rhoj is %1.5e\n", ρj)
        end
        if(ρj > eta2)
            if(debug)
                @printf("increase  ")
            end
            trrad = max(trrad, gamma*norm(dx, 1)) #for safety
        end

        if(ρj >= eta1)
            if(debug)
                @printf("update\n")
            end
            x = x + s;
            zjl = zjl + dzl;
            zju = zju + dzu;
        end

        if(rhoj < eta1)
            if(debug)
                @printf("shrink\n")
            end
            α = 1;
            while(meritFun(x + alpha*dx) > meritFun(x) + sigma*α*gradPhi'*dx)
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