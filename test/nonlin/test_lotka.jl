#In this example, we demonstrate the capacity of the algorithm to minimize a nonlinear
#model with a regularizer
function LotkaVolt()

	#so we need a model solution, a gradient, and a Hessian of the system (along with some data to fit)
	function LK(du, u, p, t)
		#p is parameter vector [I,μ, a, b, c]
		du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
		du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
	end


	u0 = [1.0; 1.0]
	tspan = (0.0, 10.0)
	savetime = .2

	#So this is all well and good, but we need a cost function and some parameters to fit. First, we take care of the parameters
	#We start by noting that the FHN model is actually the van-der-pol oscillator with some parameters set to zero
	#x' = μ(x - x^3/3 - y)
	#y' = x/μ -> here μ = 12.5
	#changing the parameters to p = [0, .08, 1.0, 0, 0]
	x0 = [1/3,1/9,2/3,1/9]
	prob_LKs = ODEProblem(LK, u0, tspan, x0)
	sol_LKs = solve(prob_LKs,reltol=1e-6, saveat=savetime)

	#also make some noie to fit later
	t = sol_LKs.t
	b = hcat(sol_LKs.u...)
	noise = .1*randn(size(b))
	data = noise + b

	# plts = plot(sol_LKs, vars=(0,1),xlabel="Time", linewidth = 4, ylabel="Species Number", label="Prey", title="LKs sol")
	# plot!(plts, sol_LKs, vars=(0,2),label="Pred", linewidth = 4)
	# plot!(plts, sol_LKs.t, data[1,:], label="Prey-data", seriestype = :scatter)
	# plot!(plts, sol_LKs.t, data[2,:], label="Pred-data", seriestype = :scatter)
	# # savefig("figs/nonlin/lotka/basic.pdf")
	# savefig(plts, "figs/nonlin/lotka/basic.tikz")
	# run(`mv figs/nonlin/lotka/basic.tex figs/nonlin/lotka/basic.tikz`)


	#so now that we have data, we want to formulate our optimization problem. This is going to be 
	#min_p ||f(p) - b||₂^2 + λ||p||₀
	#define your smooth objective function
	#First, make the function you are going to manipulate
	function Gradprob(p)
		temp_prob = remake(prob_LKs, p = p)
		temp_sol = solve(temp_prob, reltol=1e-6, saveat=savetime)
		tot_loss = 0.0
		if any((temp_sol.retcode!= :Success for s in temp_sol))
			tot_loss = Inf
		else
			temp_v = convert(Array, temp_sol)
			tot_loss = sum((temp_v - data).^2)/2
		end

		return tot_loss
	end
	function f_obj(x) #gradient and hessian info are smooth parts, m also includes nonsmooth part
		fk = Gradprob(x)
		# @show fk
		if fk==Inf 
			grad = Inf*ones(size(x))
			Hess = Inf*ones(size(x,1), size(x,1))
		else
			grad = Zygote.gradient(Gradprob, x)[1] 
			Hess = Zygote.hessian(Gradprob, x)
		end

		return fk, grad, Hess
	end

	λ = 1.0
	function h_obj(x)
		return λ*norm(x,1)
	end


	#put in your initial guesses
	xi = .5*ones(size(x0))

	(_, _, Hessapprox) = f_obj(xi)
	#all this should be unraveling in the hardproxB# code
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
	Doptions=s_options(eigmax(Hessapprox); λ=λ, verbose = 0, optTol=1e-6)


	params= IP_struct(f_obj, h_obj; FO_options = Doptions, s_alg=PG, Rkprox=prox)

	options = IP_options(;verbose=0, ϵD = 1e-5, Δk = .1, maxIter = 1000)



	#solve our problem 
	function funcF(x)
		fk = Gradprob(x)
		# @show fk
		if fk==Inf 
			grad = Inf*ones(size(x))
		else
			grad = Zygote.gradient(Gradprob, x)[1] 
		end

		return fk, grad
	end
	function proxp(z, α)
		return sign.(z).*max.(abs.(z).-(α*λ/eigmax(Hessapprox))*ones(size(z)), zeros(size(z)))
	end

	x_pr, k, Fhist, Hhist, Comp_pg = IntPt_TR(xi, params, options)


	# poptions=s_options(eigmax(Hessapprox); λ=λ, verbose = 10, optTol=1e-3)
	# xpg, xpg⁻, histpg, fevals = PGLnsch(funcF, h_obj, xi, proxp, poptions)
	popt = spg_options(;optTol=1.0e-5, progTol=1.0e-10, verbose=10,maxIter = 1000, memory=5, curvilinear=true)
	funproj(d, σ) = proxp(d, σ)
	# funproj(d) = proxp(d, .1)
	(xpg, fsave, funEvals,_,histpg) = minConf_SPG(funcF, xi, funproj, popt)

	folder = "figs/nonlin/lotka/"

	probx = remake(prob_LKs, p = x_pr)
	temp_solx = solve(probx, reltol=1e-6, saveat=savetime)
	probx = remake(prob_LKs, p = xpg)
	temp_solp = solve(probx, reltol=1e-6, saveat=savetime)



	#print out l2 norm difference and plot the two x values
	sol = hcat(sol_LKs.u...)
	solx = hcat(temp_solx.u...)
	solp = hcat(temp_solp.u...)

	fp = f_obj(x_pr)[1]
	fpt = f_obj(x0)[1]
	fpo = f_obj(xpg)[1]

	ftab = [fp, fpo, fpt]
	htab = [h_obj(x_pr)/λ, h_obj(xpg)/λ, h_obj(x0)/λ ]


	objtest = abs(fp - fpt)
	partest = norm(x_pr - x0)

	yvars = [sol[1,:], sol[2,:], solx[1,:], solx[2,:], solp[1,:], solp[2,:], data[1,:], data[2,:]]
	xvars = [t, t, t, t, t, t, t, t]
	labs = ["True-Pred", "True-Prey", "TR-Pred", "TR-Prey", "MC-Pred", "MC-Prey", "Data-Pred", "Data-Prey"]
	figen_non(xvars, yvars, labs, string(folder, "xcomp"), [" ", "Time", "Population"],2, 1)


	# hist = [Fhist + Hhist, Fhist, Hhist, histpg] 
	# labs = ["f+h: TR", "f: TR", "h: TR", "f+h: MC"]
	hist = [Fhist, histpg[1,:]]
	histx = [Array(1:length(Fhist)), histpg[2,:]] 
	labs = ["f+h: TR", "f+h: MC"]
	figen_non(histx, hist, labs, string(folder,"objcomp"), [" ", "kth Objective Evaluation", " Value "], 3, 0)
 
	figen([Comp_pg], ["TR"], string(folder,"complexity"), [" ", "kth Iteration", " Inner Prox Evaluations "], 1, 0)
	
	
	
	objtab = ftab + htab 
	vals = vcat(objtab', ftab', htab', [partest, norm(xpg - x0), 0 ]')
	pars = hcat(x0, x_pr, xpg)

	dp, df = show_table(pars, vals)
	_ = write_table(dp, df, "figs/nonlin/lotka/lotka")


	return partest, objtest

end