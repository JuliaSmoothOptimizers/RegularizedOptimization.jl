# Julia Testing function
# Generate Compressive Sensing Data
using Plots
include("bpdntable.jl")

function bpdnNoBarTrB0Binf(A, x0, b, b0, compound, k)
	#Here we just try to solve the l2-norm^2 data misfit + l1 norm regularization over the l1 trust region with -10≦x≦10
	#######
	# min_x 1/2||Ax - b||^2 + δ(λ||x||₀< k)
	m,n = size(A)
	#initialize x
	δ = k 
	λ = norm(A'*b, Inf)/100

	#define your smooth objective function
	#merit function isn't just this though right?
	function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
		r = A*x - b
		g = A'*r
		return norm(r)^2/2, g, A'*A
	end

	function h_obj(x)
		if norm(x,0) ≤ δ
			h = 0
		else
			h = Inf 
		end
		return λ*h 
	end

	#set all options
	β = eigmax(A'*A)
	Doptions=s_options(β; verbose=0, λ=λ)

	function prox(q, σ, xk, Δ)
		ProjB(w) = min.(max.(w, xk.-Δ), xk.+Δ)
		y = q + xk 
		#find largest entries
		p = sortperm(y, rev = true)
		y[p[δ+1:end]].=0 #set smallest to zero 
		y = ProjB(y)#put all entries in projection
		s = y - xk 

		# w = xk + q
		# p = sortperm(w,rev=true)
		# w[p[δ+1:end]].=0
		# s = ProjB(w) - xk
		# y = ProjB(w)
		# r = (λ/σ)*.5*((y - (xk + q)).^2 - (xk + q))
		# p = sortperm(r, rev=true)
		# y[p[δ+1:end]].=0
		# s = y - xk
		return s 
	end

	parameters = IP_struct(f_obj, h_obj; FO_options = Doptions, s_alg=PG, Rkprox=prox)
	options = IP_options(; ϵD=1e-8)
	#put in your initial guesses
	xi = ones(n,)/2

	function funcF(x)
		r = A*x - b
		g = A'*r
		return norm(r)^2, g
	end
	function proxp(z, α)
		y = zeros(size(z))
		#find largest entries
		p = sortperm(z, rev = true)
		y[p[δ+1:end]].=0 #set smallest to zero 
		return y 
	end

	x_pr, k, Fhist, Hhist, Comp_pg = IntPt_TR(xi, parameters, options)
	xpg, xpg⁻, histpg, fevals = PGLnsch(funcF, h_obj, xi, proxp, Doptions)


	folder = string("figs/bpdn/LS_B0_Binf/", compound, "/")

	fp = f_obj(x_pr)[1]+h_obj(x_pr)
	fpt =  (f_obj(x0)[1]+h_obj(x0))
	fpo =  (f_obj(xpg)[1]+h_obj(xpg))

	objtest = abs(fp - fpt)/norm(A,2)
	partest = norm(x_pr - x0)/norm(A,2)


	ftab = [f_obj(x_pr)[1], f_obj(xpg)[1], f_obj(x0)[1]]'
	htab = [h_obj(x_pr)/λ, h_obj(xpg)/λ, h_obj(x0)/λ ]'
	objtab = [fp,fpo, fpt]'
	vals = vcat(objtab, ftab, htab, [partest, norm(xpg - x0), 0 ]')
	pars = hcat(x0, x_pr, xpg)


	xvars = [x_pr, x0, xpg]; xlabs = ["TR", "True", "PG"]
	titles = ["Basis Comparison", "ith Index", " "]
	figen(xvars, xlabs, string(folder,"xcomp"), ["Basis Comparison", "ith Index", " "], 1)




	bvars = [A*x_pr, b0, A*xpg]; 
	figen(bvars, xlabs,string(folder,"bcomp"), ["Signal Comparison", "ith Index", " "], 1)
	
	
	hist = [Fhist + Hhist, Fhist, Hhist, 
			histpg] 
	labs = ["f+g: TR", "f: TR", "h: TR", "f+g: PG"]
	figen(hist, labs, string(folder,"objcomp"), ["Objective History", "kth Iteration", " Objective Value "], 3)
 
	figen([Comp_pg], "TR", string(folder,"complexity"), ["Complexity History", "kth Iteration", " Objective Function Evaluations "], 1)

	
	dp, df = show_table(pars, vals)
	_ = write_table(dp, df, string(folder,"B0binf"))



	return partest, objtest 
end
