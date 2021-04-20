@testset "Inner Descent Direction ($compound) Descent Methods" for compound = 1
	using LinearAlgebra
	using TRNC 

	m, n = compound * 25, compound * 64
	p = randperm(n)
	k = compound * 2

	# initialize x 
	x0 = zeros(n)
	p = randperm(n)[1:k]
	x0[p[1:k]] = sign.(randn(k))
	xk = 10 * randn(n)


	A, _ = qr(5 * randn((n, m)))
	B = Array(A)'
	A = Array(B)

	b0 = A * x0
	b = b0 + .005 * randn(m)
	λ = 0.1 * norm(A' * b, Inf)
	Δk = 3 * rand()


	ν = 1 / eigmax(A' * A)



	function f_obj(x) # gradient and hessian info are smooth parts, m also includes nonsmooth part
		r = A * x - b
		g = A' * r
		return norm(r)^2 / 2, g, A' * A
	end
	function h_obj(x)
		return λ * norm(x, 1)
	end

	function proxl1b2(q, σ, x, Δ) # q = s - ν*g, ν*λ, xk, Δ - > basically inputs the value you need

		ProjB(y) = min.(max.(y, q .- σ), q .+ σ)
		froot(η) = η - norm(ProjB((-x) .* (η / Δ)))
	
		# %do the 2 norm projection
		y1 = ProjB(-x) # start with eta = tau
		if (norm(y1) <= Δ)
			y = y1  # easy case
		else
			η = fzero(froot, 1e-10, Inf)
			y = ProjB((-x) .* (η / Δ))
		end
    	
		if (norm(y) <= Δ)
			snew = y
		else
			snew = Δ .* y ./ norm(y)
		end
		return snew
	end 

	(qk, ∇qk, H) = f_obj(xk)
	hk = h_obj(xk) 

	TOL = 1e-10
	MI = 5000
	Doptions = s_options(ν; maxIter=MI, verbose=0, λ=λ, optTol=TOL)
	objInner(d) = [0.5 * (d' * H * d) + ∇qk' * d + qk, H * d + ∇qk, H]



	@testset "S: l1 - B2" begin
		si⁻ = zeros(n)
		si = deepcopy(si⁻)

		s_out, s⁻_out, hispg_out, feval = PG(objInner, h_obj, zeros(n), (q, σ) -> proxl1b2(q, σ, xk, Δk), Doptions)

		# check func evals less than maxIter 
		@test feval <= MI

		# check relative accuracy 
		@test norm(s_out - s⁻_out, 2) <= TOL


		# test function outputs
		@test hispg_out[end] < qk + hk 
		
		# test for relative descent 
		@test (hispg_out[end - 1] >= hispg_out[end]) || norm(hispg_out[end - 1] - hispg_out[end - 1]) < .001
	end

	function proxl1binf(q, σ, x, Δ)
		ProjB(wp) = min.(max.(wp, q .- σ), q .+ σ)
		ProjΔ(yp) = min.(max.(yp, -Δ), Δ)
		return s = ProjΔ(ProjB(-x))
	end

	@testset "S: l1 - Binf" begin

		
		si⁻ = zeros(n)
		si = copy(si⁻)

		s_out, s⁻_out, hisf_out, feval = PG(objInner, h_obj, zeros(n), (q, σ) -> proxl1binf(q, σ, xk, Δk), Doptions)

		# check func evals less than maxIter 
		@test feval <= MI

		# check relative accuracy 
		@test norm(s_out .- s⁻_out) <= TOL
		
		# test for descent 
		@test hisf_out[end] < qk + hk 

		# test for relative descent 
		@test (hisf_out[end - 1] >= hisf_out[end]) || norm(hisf_out[end - 1] - hisf_out[end]) < .001



end

    function h_obj0(x)
	return λ * norm(x, 0)
    end

    function proxl0binf(q, σ, x, Δ)
    # @show σ/λ, λ
        c = sqrt(2 * σ)
        w = x + q
        st = zeros(size(w))

        for i = 1:length(w)
            absx = abs(w[i])
            if absx <= c
                st[i] = 0
            else
                st[i] = w[i]
            end
        end
        s = st - x
        return s 
    end

    @testset "S: l0 - Binf" begin

		s⁻ = zeros(n)
		s = copy(s⁻)
		s_out, s⁻_out, _, feval = FISTA(objInner, h_obj0, zeros(n), (q, σ) -> proxl0binf(q, σ, xk, Δk), Doptions)

		# check func evals less than maxIter 
		@test feval <= MI

		# check relative accuracy 
		@test norm(s_out .- s⁻_out) <= TOL
		
		# test for decent 
		@test f_obj(xk + s_out)[1] + h_obj(xk + s_out) < qk + hk 
		# test for relative descent 
		@test (f_obj(xk + s⁻_out)[1] + h_obj(xk + s⁻_out) >= f_obj(xk + s_out)[1] + h_obj(xk + s_out)) || norm(f_obj(xk + s⁻_out)[1] + h_obj(xk + s⁻_out) - (f_obj(xk + s_out)[1] + h_obj(xk + s_out))) < .001


end

    δ = k 
    Doptions.λ = δ
    function h_objb0(x)
	if norm(x, 0) ≤ δ
		h = 0
	else
		h = Inf 
	end
	return λ * h 
    end

    function proxB0binf(q, σ, x, Δ)
	ProjB(w) = min.(max.(w, -Δ), Δ)

	w = x + q
	p = sortperm(abs.(w), rev=true)
	w[p[δ + 1:end]] .= 0
	s = ProjB(w - x)
	# w = xk - gk
	# y = ProjB(w, zeros(size(xk)), Δ)
	# r = (1/(2*ν))*((y - (xk - gk)).^2 - (xk - gk))
	# p = sortperm(r, rev=true)
	# y[p[λ+1:end]].=0
	# s = y - xk
	return s 
    end

    @testset "S: l0 - δB0" begin # this can give you oscillatory behavior, especially in the PG! case 

		s⁻ = zeros(n)
		si = deepcopy(s⁻)
		s_out, s⁻_out, _, feval = PG(objInner, h_objb0, s⁻, (q, σ) -> proxB0binf(q, σ, xk, k), Doptions)

		# check func evals less than maxIter 
		@test feval <= MI

		
		# check relative accuracy 
		@test norm(s_out .- s⁻_out) <= TOL

		
		@test f_obj(xk + s_out)[1] + h_obj(xk + s_out) < qk + hk 


		@test f_obj(xk + s⁻_out)[1] + h_obj(xk + s⁻_out) >= f_obj(xk + s_out)[1] + h_obj(xk + s_out) 

end
end