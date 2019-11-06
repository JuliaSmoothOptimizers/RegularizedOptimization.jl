export prox_params 

mutable struct p_params
    optTol
    maxIter
    verbose
    restart
    η
    η_factor
    mod_η 
    WoptTol
    gk 
    Bk 
end

function prox_params(;optTol=1f-5, maxIter=10000, verbose=2, η =1.0, η_factor=.9, 
    gk = Vector{Float64}(undef,0), Bk = Array{Float64}(undef, 0,0))
    return p_params( grad, Hess)
end


function  prox_grad(proxp, s0, projq, options;maxiter=10000)
    """Solves descent direction s for some objective function with the structure
        min_s q_k(s) + ψ(x+s) s.t. ||s||_q⩽ σ_TR   
        for some σ_TR provided 
    Arguments
    ----------
    proxp : prox method for p-norm
        takes in z (vector), a (λ||⋅||_p), p is norm for ψ I think
    s0 : Vector{Float64,1}
        Initial guess for the descent direction
    projq : generic that projects onto ||⋅||_q⩽σ_TR norm ball 
    options : mutable structure p_params


    Returns
    -------
    s   : Vector{Float64,1}
        Final value of Algorithm 6.1 descent direction 
    w   : Vector{Float64,1}
        relaxation variable of Algorithm 6.1 descent direction 
    """
#sj,σ,ν, p_norm, q_norm, params,
    ε=options.optTol
    max_iter=options.maxIter
    ε=options.WoptTol
    #η_factor should have two values, as should η
    η = options.η
    ξ = η
    η_factor = options.η_factor
    mod_η = options.mod_η 
    gk = options.gk 
    Bk = options.Bk 
    
    if options.verbose==0
        print_freq = Inf
    elseif options.verbose==1
        print_freq = round(max_iter/10)
    elseif options.verbose==2
        print_freq = round(max_iter/100)
    else 
        print_freq = round(max_iter/200)
    end

    sj = zeros(size(s0))
    u = zeros(size(s0))
    w = zeros(size(s0))

    err=100
    w_err = norm(w - x - s)^2 
    k = 0 

    while err>ε && k<max_iter && w_err>ε_w:
        #s update
        s_ = s; 
        # prox(z,α) = prox_lp(z, α, q) #x is what is changed, z is what is put in, p is norm, a is ξ
        s = proj_prox(s_, σ, q, prox_lp)
        #w update with prox of ξ*ν*λ*||⋅||_p
        w_ = w
        wp = w - (ξ/η)*(w - x - s)
        prox_lp(w, wp, ξ*ν*λ, p_norm)
            
        w_err = norm(w - x-s)
        err = norm(s_ - s) + norm(w_ - w)
        j % print_freq ==0 && 
        @printf('p-norm: %d q-norm: %d iter: %d, w-s-x: %7.3f, err: %7.3e, η: %7.3e \n', p, q, k, w_err, err, η)
        
        if k % mod_η ==0
            η = η*η_factor
            ξ = η 
            err=100
        end 
        k = k+1
    end
 

    return s, w

end

