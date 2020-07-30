export gradient, bfgs_update, rk4Solve, OrdDiffProb



function gradient(Fcn, p; ε=1e-6)
    grad = zeros(size(p))
    p_temp = zeros(ComplexF64, size(p)) + p
    for j = 1:length(p)
        p_temp[j]+=im*ε
        grad[j] = imag(Fcn(p_temp))/ε
        p_temp[j]-=im*ε

    end

    return grad
end


function bfgs_update(Bk, sk, yk)

    f1b = sk'*Bk*sk 
    f1t = (Bk*sk)*(sk'*Bk)

    f2t = yk*yk'
    f2b = yk'*sk 

    return Bk - f1t/f1b + f2t/f2b 

end


#DiffEq wrapper
mutable struct OrdDiffSys
    ODE #function that spits out diffeq
    IC #initial conditions
    tspan #output times
    pars #parameter vector
end
function OrdDiffProb(ODE, IC, pars; tspan=[0,1]#output times
    )
    return OrdDiffSys(ODE, IC, tspan, pars)


#takes in diffeq wrapper, 
function rk4Solve(Prob; ϵ=1e-6)

    #check for increasing stepsize
    if any(diff(Prob.tspan).<0)
        # throw(ArgumentError(tspan, "tspan must be monotonically increasing."))
        sort!(Array(Prob.tspan))
    end
    t = vcat(Array(Prob.tspan[1]:ϵ:Prob.tspan[end]), Prob.tspan)
    unique!(sort!(t))
    n = length(t)


    y = zeros(n, length(Prob.IC))
    y[1,:] = Prob.IC 

    for i = 1:n-1
        tt = t[i]
        yt = y[i,:]

        k1 = Prob.ODE(yt, Prob.pars, tt)
        ymid = yt + .5 .*ϵ.*k1 

        k2 = Prob.ODE(ymid, Prob.pars, tt+ϵ*.5)
        ymid = yt + .5 .*ϵ.*k2 

        k3 = Prob.ODE(ymid, Prob.pars, tt+ϵ*.5)
        yend = yt + k3.*ϵ

        k4 = Prob.ODE(yend, Prob.pars, tt+ϵ)

        ϕ = (k1 + 2.0 .*(k2 + k3) + k4)./6
        y[i+1,:] .= yt .+ phi.*ϵ

    end

    
    return intersect(t, Prob.tspan), y

end