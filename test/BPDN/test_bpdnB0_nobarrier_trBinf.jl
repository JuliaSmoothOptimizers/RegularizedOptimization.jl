# Julia Testing function
# Generate Compressive Sensing Data
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
			h = 1
		else
			h = 2 
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
		p = sortperm(abs.(y), rev = true)
		y[p[δ+1:end]].=0 #set smallest to zero 
		y = ProjB(y)#put all entries in projection
		s = y - xk 

		return s 
	end

	parameters = IP_struct(f_obj, h_obj; FO_options = Doptions, s_alg=PG, Rkprox=prox)
	options = IP_options(; ϵD=1e-10)
	#put in your initial guesses
	xi = ones(n,)/2

	function funcF(x)
		r = A*x - b
		g = A'*r
		return norm(r)^2, g
	end
	function proxp(z, α)
		y = z
		#find largest entries
		p = sortperm(abs.(z), rev = true)
		y[p[α+1:end]].=0 #set smallest to zero
		return y 
	end

	x_pr, k, Fhist, Hhist, Comp_pg = IntPt_TR(xi, parameters, options)
	# xpg, xpg⁻, histpg, fevals = PGLnsch(funcF, h_obj, xi, proxp, Doptions)
	popt = spg_options(;optTol=1.0e-10, progTol=1.0e-10, verbose=0, memory=5, maxIter = 1000)
	# funproj(d) = oneProjector(d, 1.0, 1.0)
	funproj(d, δ) = proxp(d, δ)
	(xpg, fsave, funEvals,_,histpg) = minConf_SPG(f_obj, zeros(size(xi)), funproj, popt)


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


	xvars = [x_pr, x0, xpg]; xlabs = ["TR", "True", "MC"]
	titles = ["Basis Comparison", "ith Index", " "]
	figen(xvars, xlabs, string(folder,"xcomp"), ["Basis Comparison", "ith Index", " "], 1, 0)




	bvars = [A*x_pr, b0, A*xpg]; 
	figen(bvars, xlabs,string(folder,"bcomp"), ["Signal Comparison", "ith Index", " "], 1, 0)
	
	# hist = [Fhist + Hhist, Fhist, Hhist, histpg] 
	# labs = ["f+g: TR", "f: TR", "h: TR", "f+g: PG"]
	# figen(hist, labs, string(folder,"objcomp"), ["Objective History", "kth Iteration", " Objective Value "], 3, 1)
 	# hist = [Fhist + Hhist, histpg] 
	# labs = ["f+g: TR", "f: TR", "h: TR", "f+g: PG"]
	hist = [Fhist, histpg[1,:]]
    histx = [Array(1:length(Fhist)), histpg[2,:]] 
    labs = ["f+h: TR", "f+h: MC"]
    figen_non(histx, hist, labs, string(folder,"objcomp"), ["Objective History", "kth Objective Evaluation", " Objective Value "], 3, 0)
 
	figen([Comp_pg], ["TR"], string(folder,"complexity"), ["Complexity History", "kth Iteration", " Objective Function Evaluations "], 1, 0)

	
	dp, df = show_table(pars, vals)
	_ = write_table(dp, df, string(folder,"B0binf"))



	return partest, objtest 
end
