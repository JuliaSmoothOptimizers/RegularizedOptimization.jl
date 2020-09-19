@testset "Inner Descent Direction ($compound) Descent Methods" for compound=1
	using LinearAlgebra
	using TRNC 
	using Roots

	m,n = compound*25, compound*64
	p = randperm(n)
	k = compound*2

	#initialize x 
	x0 = zeros(n)
	p = randperm(n)[1:k]
	x0[p[1:k]]=sign.(randn(k))
	xk = 10*randn(n)


	A,_ = qr(5*randn((n,m)))
	B = Array(A)'
	A = Array(B)

	b0 = A*x0
	b = b0 + .005*randn(m)
	λ = 0.1*norm(A'*b, Inf)
	Δ = 3*rand()


	β = eigmax(A'*A)



	function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
		r = A*x - b
		g = A'*r
		return norm(r)^2/2, g, A'*A
	end
	function h_obj(x)
		return λ*norm(x,1)
	end

	function proxl1b2(q, σ) #q = s - ν*g, ν*λ, xk, Δ - > basically inputs the value you need

		ProjB(y) = min.(max.(y, q.-σ), q.+σ)
		froot(η) = η - norm(ProjB((-xk).*(η/Δ)))
	
		# %do the 2 norm projection
		y1 = ProjB(-xk) #start with eta = tau
		if (norm(y1)<= Δ)
			y = y1  # easy case
		else
			η = fzero(froot, 1e-10, Inf)
			y = ProjB((-xk).*(η/Δ))
		end
	
		if (norm(y)<=Δ)
			snew = y
		else
			snew = Δ.*y./norm(y)
		end
		return snew
	end 

	function proxl1b2!(q, σ) #q = s - ν*g, ν*λ, xk, Δ - > basically inputs the value you need

		ProjB(y) = min.(max.(y, q.-σ), q.+σ)
		froot(η) = η - norm(ProjB((-xk).*(η/Δ)))
	
		# %do the 2 norm projection
		y1 = ProjB(-xk) #start with eta = tau
		if (norm(y1)<= Δ)
			y = y1  # easy case
		else
			η = fzero(froot, 1e-10, Inf)
			y = ProjB((-xk).*(η/Δ))
		end
	
		if (norm(y)<=Δ)
			q[:] = y[:]
		else
			q[:] = Δ.*y[:]./norm(y)
		end
	end 
	(qk, ∇qk, H) = f_obj(xk)
	hk = h_obj(xk)
	Hess(d) = H*d

	TOL = 1e-10
	Doptions=s_options(β; maxIter=5000, verbose =10, λ=λ, optTol = TOL, xk = xk, Δ = Δ, ∇fk = ∇qk, Bk = A'*A)
	objInner(d) = [0.5*(d'*Hess(d)) + ∇qk'*d + qk, Hess(d) + ∇qk]

	function objInner!(d, g)
		g[:] = ∇qk[:]
		g += Hess(d)
		return 0.5*(d'*Hess(d)) + ∇qk'*d + qk
	end



	@testset "S: l1 - B2" begin
		s⁻ = zeros(n)
		s = deepcopy(s⁻)

		s_out, s⁻_out, hispg_out, feval = PG(objInner,h_obj, zeros(n), proxl1b2, Doptions)
		s⁻, hispg, fevals = PG!(objInner!, h_obj, s⁻, proxl1b2!, Doptions)

		#check func evals less than maxIter 
		@test feval <= 5000
		@test fevals <= 5000

		#check relative accuracy 
		@test norm(s_out - s⁻_out, 2) <= TOL
		@test norm(s - s⁻, 2) <= TOL
		@show s- s⁻


		#test function outputs
		@test hispg_out[end] <= qk + hk 
		@test hispg[end] <= qk + hk #check for decrease 
		
		#test for relative descent 
		@test (hispg_out[end-1] >= hispg_out[end]) || norm(hispg_out[end-1] - hispg_out[end-1])<.001
		@test (hispg[end-1] >=hispg[end]) || norm(hispg[end-1] - hispg[end])<.001

	end

	function proxl1binf(q, σ)
		Fcn(yp) = (yp-xk-q).^2/2+σ*abs.(yp)
		ProjB(wp) = min.(max.(wp,xk.-Δ), xk.+Δ)
		
		y1 = zeros(size(xk))
		f1 = Fcn(y1)
		idx = (y1.<xk.-Δ) .| (y1.>xk .+ Δ) #actually do outward since more efficient
		f1[idx] .= Inf

		y2 = ProjB(xk+q.-σ)
		f2 = Fcn(y2)
		y3 = ProjB(xk+q.+σ)
		f3 = Fcn(y3)

		smat = hcat(y1, y2, y3) #to get dimensions right
		fvec = hcat(f1, f2, f3)

		f = minimum(fvec, dims=2)
		idx = argmin(fvec, dims=2)
		s = smat[idx]-xk

		return dropdims(s, dims=2)
	end
	function proxl1binf!(q, σ)
		Fcn(yp) = (yp-xk-q).^2/2+σ*abs.(yp)
		ProjB(wp) = min.(max.(wp,xk.-Δ), xk.+Δ)
		
		y1 = zeros(size(xk))
		f1 = Fcn(y1)
		idx = (y1.<xk.-Δ) .| (y1.>xk .+ Δ) #actually do outward since more efficient
		f1[idx] .= Inf
	
		y2 = ProjB(xk+q.-σ)
		f2 = Fcn(y2)
		y3 = ProjB(xk+q.+σ)
		f3 = Fcn(y3)

		smat = hcat(y1, y2, y3) #to get dimensions right
		fvec = hcat(f1, f2, f3)
	
		f = minimum(fvec, dims=2)
		idx = argmin(fvec, dims=2)
		s = smat[idx]-xk
		s = dropdims(s, dims=2)
		q[:] = s[:]
	end

	@testset "S: l1 - Binf" begin

		
		s⁻ = zeros(n)
		s = copy(s⁻)

		s_out, s⁻_out, _, feval = FISTA(objInner,h_obj, zeros(n), proxl1binf, Doptions)
		s⁻, _, fevals = FISTA!(objInner!,h_obj, s⁻, proxl1binf!, Doptions)

		#check func evals less than maxIter 
		@test feval <= 5000
		@test fevals <= 5000

		#check relative accuracy 
		@test norm(s_out .- s⁻_out) <= TOL
		@test norm(s .- s⁻) <= TOL
		
		#test for descent 
		@test f_obj(xk+s_out)[1]+h_obj(xk+s_out) < qk + hk 
		@test f_obj(xk+s)[1]+h_obj(xk+s) < qk + hk 

		#test for relative descent 
		@test (f_obj(xk+s⁻_out)[1]+h_obj(xk+s⁻_out) >= f_obj(xk+s_out)[1]+h_obj(xk+s_out)) || norm(f_obj(xk+s⁻_out)[1]+h_obj(xk+s⁻_out) - (f_obj(xk+s_out)[1]+h_obj(xk+s_out)))<.001
		@test (f_obj(xk+s⁻)[1]+h_obj(xk+s⁻) >= f_obj(xk+s)[1]+h_obj(xk+s)) || norm(f_obj(xk+s⁻)[1]+h_obj(xk+s⁻) - (f_obj(xk+s)[1]+h_obj(xk+s)))<.001


end

function h_obj0(x)
	return λ*norm(x, 0)
end

function proxl0binf(q, σ)

	ProjB(y) = min.(max.(y, xk.-Δ),xk.+Δ) # define outside? 
	c = sqrt(2*σ)
	w = xk+q
	st = zeros(size(w))

	for i = 1:length(w)
		absx = abs(w[i])
		if absx <=c
			st[i] = 0
		else
			st[i] = w[i]
		end
	end

	s = ProjB(st) - xk

	return s 
end 

function proxl0binf!(q, σ)

	ProjB(y) = min.(max.(y, xk.-Δ),xk.+Δ) # define outside? 
	c = sqrt(2*σ)
	q[:] += xk[:]

	for i = 1:length(q)
		absx = abs(q[i])
		if absx <=c
			q[i] = 0
		end
	end
	q[:] = ProjB(q)[:]
	q[:] -= xk[:] 
end 

@testset "S: l0 - Binf" begin

		
		s⁻ = zeros(n)
		s = copy(s⁻)
		s_out, s⁻_out, _, feval = FISTA(objInner, h_obj0, zeros(n), proxl0binf, Doptions)
		s⁻, _, fevals = FISTA!(objInner!,h_obj0, s, proxl0binf!, Doptions)

		#check func evals less than maxIter 
		@test feval <= 5000
		@test fevals <= 5000

		
		#check relative accuracy 
		@test norm(s_out .- s⁻_out) <= TOL
		@test norm(s .- s⁻) <= TOL
		
		#test for decent 
		@test f_obj(xk+s_out)[1]+h_obj(xk+s_out) < qk + hk 
		@test f_obj(xk+s)[1]+h_obj(xk+s) < qk + hk 
		#test for relative descent 
		@test (f_obj(xk+s⁻_out)[1]+h_obj(xk+s⁻_out) >= f_obj(xk+s_out)[1]+h_obj(xk+s_out)) || norm(f_obj(xk+s⁻_out)[1]+h_obj(xk+s⁻_out) - (f_obj(xk+s_out)[1]+h_obj(xk+s_out)))<.001
		@test (f_obj(xk+s⁻)[1]+h_obj(xk+s⁻) >= f_obj(xk+s)[1]+h_obj(xk+s)) || norm(f_obj(xk+s⁻)[1]+h_obj(xk+s⁻) - (f_obj(xk+s)[1]+h_obj(xk+s)))<.001


end

# δ = k 
# Doptions.λ = δ
# function h_objb0(x)
# 	if norm(x,0) ≤ δ
# 		h = 0
# 	else
# 		h = Inf 
# 	end
# 	return λ*h 
# end

# function proxB0binf(q, σ)
# 	ProjB(w) = min.(max.(w, -Δ), Δ)

# 	w = xk - q
# 	p = sortperm(w,rev=true)
# 	w[p[δ+1:end]].=0
# 	s = ProjB(w) - xk
# 	# w = xk - gk
# 	# y = ProjB(w, zeros(size(xk)), Δ)
# 	# r = (1/(2*ν))*((y - (xk - gk)).^2 - (xk - gk))
# 	# p = sortperm(r, rev=true)
# 	# y[p[λ+1:end]].=0
# 	# s = y - xk
# 	return s 
# end

# function proxB0binf!(q, σ)
# 	ProjB(y) = min.(max.(y, -Δ), Δ)

# 	w = xk - q
# 	pp = sortperm(w,rev=true)
# 	w[pp[δ+1:end]].=0
# 	w = ProjB(w) - xk

# 	q[:] = w[:]
# end

# @testset "S: l0 - δB0" begin #this can give you oscillatory behavior, especially in the PG! case 

# 		s⁻ = zeros(n)
# 		s = deepcopy(s⁻)
# 		s_out, s⁻_out, _, feval = PG(objInner,h_objb0, s⁻, proxB0binf, Doptions)

# 		#check func evals less than maxIter 
# 		@test feval <= 5000

		
# 		#check relative accuracy 
# 		@test norm(s_out .- s⁻_out) <= TOL
# 		#problem is nonconvex
# 		# @test norm(s .- s⁻) <= TOL || norm(s - s_out)<TOL 
		
# 		@test f_obj(xk+s_out)[1]+h_obj(xk+s_out) < qk + hk 
# 		@test f_obj(xk+s⁻_out)[1]+h_obj(xk+s⁻_out) >= f_obj(xk+s_out)[1]+h_obj(xk+s_out) 

# end
end