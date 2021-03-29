
include("Nonlin_table.jl")

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
		grad = Zygote.gradient(Gradprob, x)[1] 
		# Hess = Zygote.hessian(Gradprob, x)
	end
	return fk, grad
end


TOL = 1e-6
MI = 500
ϵ = 1e-3
λ = 1.0 
# set all options
Doptions = s_options(1.0; λ=λ, optTol=TOL, verbose=0, maxIter=1000)
options = TRNCoptions(;verbose=10, ϵD=ϵ, maxIter=MI, β=1e16)

# this is for l0 norm 
function h_obj(x)
	return norm(x, 0) 
end
function proxtr(q, σ, xk, Δ)
	ProjB(y) = min.(max.(y, -Δ), Δ) # define outside? 
	# @show σ/λ, λ
	c = sqrt(2 * σ)
	w = xk + q
	st = zeros(size(w))

	for i = 1:length(w)
		absx = abs(w[i])
		if absx <= c
			st[i] = 0
		else
			st[i] = w[i]
		end
	end
	s = ProjB(st - xk)
	return s 
end

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

params = TRNCstruct(f_obj, h_obj, λ; FO_options=Doptions, s_alg=PG, ψχprox=proxtr, χk=(s) -> norm(s, Inf), HessApprox=LSR1Operator)
xi = ones(size(pars_FH))


@info "running TR with our own objective"
xtr, ktr, Fhisttr, Hhisttr, Comp_pg = TR(xi, params, options)
proxnum = [0, sum(Comp_pg)]

Ival = f_obj(xi)[1] + λ * h_obj(xi)

solverp = ProximalAlgorithms.PANOC(tol=ϵ, verbose=true, freq=1, maxit=MI)
ϕ = LeastSquaresObjective(f_obj, h_obj, 0, [Ival])
g = ProxOp(h_obj, proxl0, λ, 0)

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
xlabs = ["True", "TR", "PANOC", "ZFP"]

hist = [Fhisttr + Hhisttr, histpanoc, histz]
fig_preproc(xvars, xlabs, hist,[Comp_pg], Aval, folder)

vals, pars = tab_preproc(f_obj, h_obj, xvars, proxnum, hist, Aval, λ)

dp, df = show_table(pars, vals, xlabs)
_ = write_table(dp, df, string(folder, "fhl0"))


@info "running QR"
# xtr, k, Fhist, Hhist, Comp_pg = TR(xi, params, options)

function proxl0s(q, σ, xk, Δ)
	# @show σ/λ, λ
	c = sqrt(2 * σ)
	w = xk + q
	st = zeros(size(w))

	for i = 1:length(w)
		absx = abs(w[i])
		if absx <= c
			st[i] = 0
		else
			st[i] = w[i]
		end
	end
	s = st - xk
	return s 
end


folder = "figs/nonlin/FH/pg_l0/"
parametersQR = TRNCstruct(f_obj, h_obj, λ; FO_options=Doptions, ψχprox=proxl0s, χk=(s) -> norm(s, Inf))
optionsQR = TRNCoptions(; σk=1e4, ϵD=TOL, verbose=1, maxIter = MI*10) # options, such as printing (same as above), tolerance, γ, σ, τ, w/e

# input initial guess
xqr, kqr, Fhistqr, Hhistqr, Comp_pgqr = QuadReg(ones(size(xi)), parametersQR, optionsQR)


@info "running FBS with our objective"
ϕ.count = 0
ϕ.hist = [Ival]
g.count = 0
solverpg = ProximalAlgorithms.ForwardBackward(tol=ϵ, verbose=true, freq=1, maxit=10)
xpg, kpg = my_fbs(solverpg, ones(size(xi2)), f=ϕ, g=g)
Histpg = ϕ.hist 

xvars = [x0, xqr, xpg]
xlabs = ["True", "QR", "PG"]
histp = [Fhistqr+Hhistqr, Histpg]
vals, pars = tab_preproc(f_obj, h_obj, xvars,[0,sum(Comp_pgqr), g.count],  histp, Aval, λ)

fig_preproc(xvars,xlabs, histp,[Comp_pgqr], Aval, folder)

dp, df = show_table(pars, vals, xlabs)
_ = write_table(dp, df, string(folder, "fhl0_qr"))
# 	@info "Fitzhugh-Nagumo to Van-der-Pol: ||F(p) - b||² + χ_{||p||₀≤δ}; ||⋅||_∞  ≤Δ"