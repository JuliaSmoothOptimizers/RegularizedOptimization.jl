
include("Nonlin_table.jl")

function nonlintests()
	# so we need a model solution, a gradient, and a Hessian of the system (along with some data to fit)
	function FH_ODE(dx, x, p, t)
		# p is parameter vector [I,μ, a, b, c]
		V, W = x 
		I, μ, a, b, c = p
		dx[1] = (V - V^3 / 3 -  W + I) / μ
		dx[2] = μ * (a * V - b * W + c)
	end


	u0 = [2.0; 0.0]
	tspan = (0.0, 20.0)
	savetime = .2

	pars_FH = [0.5, 0.08, 1.0, 0.8, 0.7]
	prob_FH = ODEProblem(FH_ODE, u0, tspan, pars_FH)


	# So this is all well and good, but we need a cost function and some parameters to fit. First, we take care of the parameters
	# We start by noting that the FHN model is actually the van-der-pol oscillator with some parameters set to zero
	# x' = μ(x - x^3/3 - y)
	# y' = x/μ -> here μ = 12.5
	# changing the parameters to p = [0, .08, 1.0, 0, 0]
	x0 = [0, .2, 1.0, 0, 0]
	prob_VDP = ODEProblem(FH_ODE, u0, tspan, x0)
	sol_VDP = solve(prob_VDP, reltol=1e-6, saveat=savetime)


	# also make some noie to fit later
	t = sol_VDP.t
	b = hcat(sol_VDP.u...)
	noise = .1 * randn(size(b))
	data = noise + b

	# so now that we have data, we want to formulate our optimization problem. This is going to be 
	# min_p ||f(p) - b||₂^2 + λ||p||₀
	# define your smooth objective function
	# First, make the function you are going to manipulate
	function Gradprob(p)
		temp_prob = remake(prob_FH, p=p)
		temp_sol = solve(temp_prob, Vern9(), abstol=1e-14, reltol=1e-14, saveat=savetime, verbose=false)
		tot_loss = 0.

		if any((temp_sol.retcode != :Success for s in temp_sol))
			tot_loss = Inf
		else
			temp_v = convert(Array, temp_sol)
			tot_loss = sum(((temp_v - data).^2) ./ 2)
		end
		return tot_loss
	end
	function f_obj(x) # gradient and hessian info are smooth parts, m also includes nonsmooth part
		fk = Gradprob(x)
		if fk == Inf 
			grad = Inf * ones(size(x))
			# Hess = Inf*ones(size(x,1), size(x,1))
		else
			grad = ForwardDiff.gradient(Gradprob, x)
			# Hess = ForwardDiff.hessian(Gradprob, x)
		end
		return fk, grad
	end

	MI = 500
	ϵ = 1e-3
	λ = 1.0 
	# set all options
	options = TRNCoptions(;verbose=10, ϵ=ϵ, maxIter=MI, β=1e16)
	subopts = TRNCoptions(; maxIter = 5000)

	function Aval(x, ξ)
		if ξ == 0
			sol = data
		else
			sol = solve(remake(prob_FH, p=x), Vern9(), abstol=1e-14, reltol=1e-14, saveat=savetime, verbose=false)
		end
		return sol 
	end

	function proxl0(z, α)
		y = zeros(size(z))
		for i = 1:length(z)
			if abs(z[i]) > sqrt(2 * α * λ)
				y[i] = z[i]
			end
		end
		return y
	end

	@info "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + λ||p||₀; ||⋅||_∞  ≤Δ"
	folder = "figs/nonlin/FH/quad_l0/"

	χ = NormLinf(1.0)
	xi = ones(size(pars_FH))
	ϕt = LBFGSModel(ADNLPModel(Gradprob, xi))
	h = NormL0(λ)


	@info "running TR-PG with our own objective"
	xtr, ktr, Fhisttr, Hhisttr, Comp_pg = TR(ϕt, h, χ, options; subsolver_options = subopts, s_alg=PG)
	proxnum = [0, sum(Comp_pg[2, :])]
	Ival = obj(ϕt, xi) + h(xi)

	solverp = ProximalAlgorithms.PANOC(tol=ϵ, verbose=true, freq=1, maxit=MI)
	ϕ = LeastSquaresObjective(f_obj, (x)->λ*norm(x, 0), 0, [Ival])
	g = ProxOp((x)->λ*norm(x, 0), proxl0, 0)

	xi2 = copy(xi)


	@info "running PANOC with our own objective"
	xpanoc, kpanoc = my_panoc(solverp, xi, f=ϕ, g=g)
	histpanoc = ϕ.hist
	append!(proxnum, g.count)

	@info "running ZeroFPR with our own objective"
	ϕ.count = 0
	ϕ.hist = [Ival]
	g.count = 0
	solverz = ProximalAlgorithms.ZeroFPR(tol=ϵ, verbose=true, freq=1, maxit=MI)
	xz, kz = my_zerofpr(solverz, xi2, f=ϕ, g=g)
	histz = ϕ.hist 
	append!(proxnum, g.count)


	xvars = [x0, xtr, xpanoc, xz]
	xlabs = ["True", "TR-PG", "PANOC", "ZFP"]

	hist = [ktr, histpanoc, histz]
	fig_preproc(xvars, xlabs, hist,[Comp_pg[2,:]], Aval, folder)

	vals, pars = tab_preproc(ϕt, h, xvars, proxnum, hist, Aval, λ)

	dp, df = show_table(pars, vals, xlabs)
	_ = write_table(dp, df, string(folder, "fhl0"))


	@info "running R2"
	folder = "figs/nonlin/FH/pg_l0/"
	optionsQR = TRNCoptions(; ϵ=ϵ, verbose=1, maxIter = MI*10) # options, such as printing (same as above), tolerance, γ, σ, τ, w/e

	# input initial guess
	xqr, kqr, Fhistqr, Hhistqr, Comp_pgqr = QRalg(ϕt, h, optionsQR; x0 = ones(size(xi)))


	@info "running FBS with our objective"
	ϕ.count = 0
	ϕ.hist = [Ival]
	g.count = 0
	solverpg = ProximalAlgorithms.ForwardBackward(tol=ϵ, verbose=true, freq=1, maxit=MI*10)
	xpg, kpg = my_fbs(solverpg, ones(size(xi2)), f=ϕ, g=g)
	Histpg = ϕ.hist 

	xvars = [x0, xqr, xpg]
	xlabs = ["True", "R2", "PG"]
	histp = [kqr, Histpg]
	vals, pars = tab_preproc(ϕt, h, xvars,[0,sum(Comp_pgqr), g.count],  histp, Aval, λ)

	fig_preproc(xvars,xlabs, histp,[Comp_pgqr], Aval, folder)

	dp, df = show_table(pars, vals, xlabs)
	_ = write_table(dp, df, string(folder, "fhl0_qr"))
	# @info "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + χ_{||p||₀≤δ}; ||⋅||_∞  ≤Δ"

	@info "running TR-R2 with our own objective"
	xi = ones(size(pars_FH))
	ϕt = LBFGSModel(ADNLPModel(Gradprob, xi))
	xtrqr, ktrqr, Fhisttrqr, Hhisttrqr, Comp_pgtrqr = TR(ϕt, h, χ, options; subsolver_options = subopts)
	@show length(ktrqr), norm(xtrqr - x0), sum(Comp_pgtrqr[2,:])
	@show xtrqr, Fhisttrqr[end], Hhisttrqr[end], x0

end