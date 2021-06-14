export prox_split_1w, prox_split_2w

"""Solves descent direction s for some objective function with the structure
	min_s q_k(s) + ψ(x+s) s.t. ||s||_q⩽ Δ
	for some Δ provided
Arguments
----------
proxp : prox method for p-norm
	takes in z (vector), a (λ||⋅||_p), p is norm for ψ I think
s0 : Vector{Float64,1}
	Initial guess for the descent direction
projq : generic that projects onto ||⋅||_q⩽Δ norm ball
options : mutable structure p_params


Returns
-------
s   : Vector{Float64,1}
	Final value of Algorithm 6.1 descent direction
w   : Vector{Float64,1}
	relaxation variable of Algorithm 6.1 descent direction
"""
function  prox_split_1w(proxp, s0, projq, options)
    ε=options.optTol
    max_iter=options.maxIter
    ε_w=options.WoptTol
    λ = options.λ
    Δ = options.Δ
    restart = options.restart
    #ν_factor should have two values, as should ν
    ν = options.ν
    ξ = options.β^(-1)
    η_factor = options.η_factor
    ∇fk = options.∇fk
    Bk = options.Bk
    xk = options.xk

    if options.verbose==0
        print_freq = Inf
    elseif options.verbose==1
        print_freq = round(max_iter/10)
    elseif options.verbose==2
        print_freq = round(max_iter/100)
    else
        print_freq = 1
    end

    s = zeros(size(s0))
    u = s0+xk
    w = zeros(size(s0))

    err=100
    w_err = norm(w - xk - s)^2
    k = 1

    for i=1:restart
    while err>ε && k<max_iter && w_err>ε_w
        #store previous values
        s_ = s
        u_ = u
        w_ = w
        # prox(z,α) = prox_lp(z, α, q) #x is what is changed, z is what is put in, p is norm, a is ξ
        #I don't really like the Prox list that's currently in ProxLQ so I might make my own

        #u update with prox_(ξ*λ*||⋅||_p)(...)
        u = proxp(u_ - ξ*(∇fk +Bk*(u_ - xk) - (w_ - u_ + xk)/η), ξ*λ)

        #update s
        s = u - xk

        #w update
        w = projq(s, Δ)

        w_err = norm(w - xk-s)^2
        err = norm(s_ - s) + norm(w_ - w)
        k % print_freq ==0 && @printf("Iter: %4d, ||w-x-s||^2: %1.5e, s-err: %1.5e, w-err: %1.5e η: %1.5e, ξ: %1.5e \n", k, w_err, norm(s_ - s), norm(w_ - w), η, ξ)

        k = k+1
    end
            η *=η_factor
            err=100
            k=0
	end

    return s,s_, w
end

"""Solves descent direction s for some objective function with the structure
	min_s q_k(s) + ψ(x+s) s.t. ||s||_q⩽ Δ
	for some Δ provided
Arguments
----------
proxp : prox method for p-norm
	takes in z (vector), a (λ||⋅||_p), p is norm for ψ I think
s0 : Vector{Float64,1}
	Initial guess for the descent direction
projq : generic that projects onto ||⋅||_q⩽Δ norm ball
options : mutable structure p_params


Returns
-------
s   : Vector{Float64,1}
	Final value of Algorithm 6.2 descent direction
w   : Vector{Float64,1}
	relaxation variable of Algorithm 6.2 descent direction
"""
function  prox_split_2w(proxp, s0, projq, options)

    ε=options.optTol
    max_iter=options.maxIter
    ε_w=options.WoptTol
    λ = options.λ
    Δ = options.Δ
    restart = options.restart
    #η_factor should have two values, as should η
    η = options.η
    ξ = η
    η_factor = options.η_factor
    ∇fk = options.∇fk
    Bk = options.Bk
    xk = options.xk

	print_freq = options.verbose

    u = s0+xk
	u_ = copy(u)
    w1 = u #zeros(size(s0))
    w2 = u-xk #zeros(size(s0))

    err=100
    w1_err = norm(w1 - u)^2
    w2_err = norm(w2 - u + xk)^2
	s_feas = norm(s0,1)-Δ
    k = 1
    b_pt1 = Bk*xk - ∇fk

    for i=1:restart
		A = Bk + (2/η)*I(size(Bk,1))
    while err>ε && k<max_iter #&& (w1_err+w2_err)>ε_w

        #store previous values
        u_ = u
        w1_ = w1
        w2_ = w2
        # prox(z,α) = prox_lp(z, α, q) #x is what is changed, z is what is put in, p is norm, a is ξ
        #I don't really like the Prox list that's currently in ProxLQ so I might make my own

        #u update with prox_(ξ*λ*||⋅||_p)(...)
        # u = cg(A, b_pt1 + (w2+xk+w1)/η; maxiter=5)
        u = fastcg(A,u_, b_pt1 + (w2+xk+w1)/η; maxiter=100)


        #update w1
        w1 = proxp(u, ξ*λ)

        #w2 update
        w2 = projq(u - xk, Δ)

        w1_err = norm(w1 - u)^2
        w2_err = norm(w2 - u + xk)^2
        err = norm(u_ - u) + norm(w1_ - w1) + norm(w2_ - w2)
        s_feas = norm(u-xk, 1)-Δ
        k % print_freq ==0 &&
        @printf("iter: %d, ||w1-u||²: %7.3e, ||w2-u+xk||²: %7.3e, err: %7.3e, η: %7.3e, s_feas: %7.3e, ||w1||_1: %7.3e, ||w2||_1: %7.3e, u-sum: %7.3e\n", k, w1_err, w2_err,
		 err, η, s_feas, norm(w1,1), norm(w2,1), ∇fk'*(u-xk)+ 1/2*(u-xk)'*Bk*(u-xk))

        k = k+1
		# η = η*η_factor
		# ξ = η
    end
            η = η*η_factor
            ξ = η
            err=100
            k=1
    end

    return u - xk, u_ - xk, s_feas, err

end

function fastcg(A, x, b; epsilon=1f-5, maxiter=size(b,1))
	r = b - A*x
	p = r
	rsold = r'*r
	for i = 1:maxiter
	    Ap = A*p
	    alpha = rsold/(p'*Ap)
	    x = x+alpha*p
	    r = r-alpha*Ap
	    rsnew = r'*r
	    if sqrt(rsnew)<epsilon
	        break
	    end
	    p = r+(rsnew/rsold)*p
	    rsold = rsnew
	end
return x

end