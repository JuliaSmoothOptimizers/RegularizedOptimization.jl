include("ProxLQ.jl")
include("ProxProj.jl")
using LinearAlgebra, Printf

export prox_params, prox_grad 

mutable struct p_params
    err 
    η_ω
    η_ω_factor
    iter_crit
    stop_crit
    σ
    λ
    z
    k 
    converged 
    printevery 
    epsilon
end

function prox_params(;err = 100, η_ω =1.0, η_ω_factor=.9, iter_crit=10, stop_crit=1000, λ=1.0, σ = 1.0,x = Vector{Float64}(undef,0), 
 k = 0.0,  printevery=20, epsilon=1e-6)
    return p_params( grad, Hess)
end


function  prox_grad(sj,σ,ν, p_norm, q_norm, params)
#η_factor should have two values, as should η
err = params.err
η_ω = params.η_ω
η_ω_factor = params.η_ω_factor
iter_crit = params.iter_crit
stop_crit = params.stop_crit   
x = params.x 
λ= params.λ
printevery = params.printevery 
x_switch = params.x_switch 
ω_err = norm(ω - x - s)^2 

#initialize the rest of the algorithm
epsilon = params.epsilon; 
count_total = 0;   

for i = 1:iter_crit
    ξ = η_ω
   while err>converged && count_total<stop_crit:
            #s update
            s_ = s; 
            # prox(z,α) = prox_lp(z, α, q) #x is what is changed, z is what is put in, p is norm, a is ξ
            s = proj_prox(s_, σ, q, prox_lp)
            #ω update with prox of ξ*ν*λ*||⋅||_p
            ω_ = ω; 
            ωp = ω - (ξ/η_ω)*(ω - x - s)
            prox_lp(ω, ωp, ξ*ν*λ, p_norm)
            

            ω_err = norm(ω - x-s)
            err = norm(s_ - s) + norm(ω_ - ω)
            j % ptf ==0 && 
            @printf('p-norm: %d q-norm: %d iter: %d, ω-s-x: %7.3f, err: %7.3e, η: %7.3e \n', p, q, count_total, ω_err, err, η_ω)
            
            count_total = count_total+1
   end
   η_ω = η_ω*η_ω_factor;
   count_total = 0; 
   err = 100; 
end
 

return s, ω

end


function prox_quad(objInner, s0, funProj, tr_options;maxiter=10000)

    
    sj = zeros(size(s0))

    for i = 1:maxiter 
        z = 



    end

return s 
end
