# Julia Testing function
# Generate Compressive Sensing Data
include("bpdntable.jl")

function bpdnNoBarTrBinf(A, x0, b, b0, compound)

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

	#all this should be unraveling in the hardproxB# code
	function prox(q, σ, xk, Δ)
		# Fcn(yp) = (yp-xk-q).^2/(2*ν)+λ*abs.(yp)
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

	#set all options
	β = eigmax(A'*A)
	Doptions=s_options(β; maxIter=10000, λ=λ, verbose=0)



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
		return sign.(z).*max.(abs.(z).-(α)*ones(size(z)), zeros(size(z)))
	end


	x_pr, k, Fhist, Hhist, Comp_pg = IntPt_TR(xi, parameters, options)
	xpg, xpg⁻, histpg, fevals = PGLnsch(funcF, h_obj, xi, proxp, Doptions)


	folder = string("figs/bpdn/LS_l1_Binf/", compound, "/")

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
	
	
	hist = [Fhist + zeros(size(Fhist)), Fhist, zeros(size(Fhist)), 
			histpg, histpg, zeros(size(histpg))] 
	labs = ["f+g: TR", "f: TR", "h: TR", "f+g: PG", "f: PG", "h: PG"]
	figen(hist, labs, string(folder,"objcomp"), ["Objective History", "kth Iteration", " Objective Value "], 3)
 
	figen([Comp_pg], "TR", string(folder,"complexity"), ["Complexity History", "kth Iteration", " Objective Function Evaluations "], 1)

	
	dp, df = show_table(pars, vals)
	_ = write_table(dp, df, string(folder,"l1binf"))



	return partest, objtest 
end
