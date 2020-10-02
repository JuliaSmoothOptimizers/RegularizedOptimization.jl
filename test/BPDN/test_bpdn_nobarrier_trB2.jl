# Julia Testing function
# Generate Compressive Sensing Data
include("bpdntable.jl")
#Here we just try to solve the l2-norm^2 data misfit + l1 norm regularization over the l1 trust region with -10≦x≦10
#######
function bpdnNoBarTrB2(A, x0, b, b0, compound)

#Here we just try to solve the l2-norm^2 data misfit + l1 norm regularization over the l1 trust region with -10≦x≦10
	#######
	# min_x 1/2||Ax - b||^2 + λ||x||₁
	m,n = size(A)
	λ = norm(A'*b, Inf)/100

	#define your smooth objective function
	#merit function isn't just this though right?
	function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
		r = A*x - b
		g = A'*r
		return norm(r)^2/2, g, A'*A
	end

	function h_obj(x)
		return λ*norm(x,1) #, g∈∂h
	end

	function prox(q, σ, xk, Δ) #q = s - ν*g, ν*λ, xk, Δ - > basically inputs the value you need

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

	#set all options
	β = eigmax(A'*A)
	Doptions=s_options(β;maxIter=1000, verbose =0, λ=λ)


	parameters = IP_struct(f_obj, h_obj; FO_options = Doptions, s_alg=FISTA, Rkprox=prox)
	options = IP_options(;ϵD = 1e-10)
	#put in your initial guesses
	xi = ones(n,)/2

	function funcF(x)
		r = A*x - b
		g = A'*r
		return norm(r)^2, g
	end
	function proxp(z, α)
		return sign.(z).*max.(abs.(z).-(α)*ones(size(z)), zeros(size(z)))
	end

	x_pr, k, Fhist, Hhist, Comp_pg = IntPt_TR(xi, parameters, options)
	xpg, xpg⁻, histpg, fevals = PGLnsch(funcF, h_obj, xi, proxp, Doptions)

	folder = string("figs/bpdn/LS_l1_B2/", compound, "/")

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
	figen(xvars, xlabs, string(folder,"xcomp"), ["Basis Comparison", "ith Index", " "], 1, 0)




	bvars = [A*x_pr, b0, A*xpg]; 
	figen(bvars, xlabs,string(folder,"bcomp"), ["Signal Comparison", "ith Index", " "], 1, 0)


	hist = [Fhist + Hhist, Fhist, Hhist, 
		histpg] 
	labs = ["f+g: TR", "f: TR", "h: TR", "f+g: PG"]
	figen(hist, labs, string(folder,"objcomp"), ["Objective History", "kth Iteration", " Objective Value "], 3, 1)

	figen([Comp_pg], "TR", string(folder,"complexity"), ["Complexity History", "kth Iteration", " Objective Function Evaluations "], 1, 1)


	dp, df = show_table(pars, vals)
	_ = write_table(dp, df, string(folder,"l1b2"))



	return partest, objtest 
end
