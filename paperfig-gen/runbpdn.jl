
include("Lin_table.jl")

compound = 1
m, n = compound * 200, compound * 512
k = compound * 10
A = 5 * randn(m, n)
x0  = rand(n, )
xi = zeros(size(x0))
b0 = A * x0
α = .01
b = b0 + α * randn(m, )

function f_obj(x)
	f = .5 * norm(A * x - b)^2
	g = A' * (A * x - b)
	return f, g
end
function h_obj(x)
	return 0
end
m, n = size(A)
MI = 1000
TOL = 1e-6
λ = 1.0 
# set all options
Doptions = s_options(1 / eigmax(A' * A); optTol=TOL, maxIter=1000, verbose=0)
options = TRNCoptions(;verbose=0, ϵD=TOL, maxIter=MI)
parameters = TRNCstruct(f_obj, h_obj, λ; FO_options=Doptions, s_alg=PG, HessApprox=LSR1Operator)
solverp = ProximalAlgorithms.PANOC(tol=TOL, verbose=true, freq=1, maxit=MI)
solverz = ProximalAlgorithms.ZeroFPR(tol=TOL, verbose=true, freq=1, maxit=MI)

@info "running LS bfgs, h=0"
folder = string("figs/ls_bfgs/", compound, "/")
xi = zeros(size(x0))

function h_objmod(x)
	if norm(x, 2) ≤ 100
		h = 0
	else
		h = Inf
	end
	return h 
end
function tr_norm(z, σ, x, Δ)
	return z ./ max(1, norm(z, 2) / Δ)
end

# g = IndBallL2(100)
ϕ = LeastSquaresObjective((z) -> [norm(A * z - b)^2, A' * (A * z - b)], (z) -> 0, 0, [])
g = ProxOp(h_objmod, (z, σ) -> tr_norm(z, σ, 1, 100), 1, 0)
parameters.χk = (s) -> norm(s, 2)
parameters.ψχprox = tr_norm 
_, _ = evalwrapper(x0, xi, A, f_obj, h_obj,ϕ, g, λ, parameters, options, solverp,solverz, folder)



@info "running LS bfgs, h=l1, tr = linf"
# start bpdn stuff 
x0 = zeros(n)
p   = randperm(n)[1:k]
x0 = zeros(n, )
x0[p[1:k]] = sign.(randn(k))

A, _ = qr(randn(n, m))
B = Array(A)'
A = Array(B)

b0 = A * x0
b = b0 + α * randn(m, )
λ = norm(A' * b, Inf) / 100 # this can change around 

function f_obj(x)
	f = .5 * norm(A * x - b)^2
	g = A' * (A * x - b)
	return f, g
end
ϕ.smooth = f_obj



Doptions.λ = λ
xi = zeros(size(x0))
folder = string("figs/bpdn/LS_l1_Binf/", compound, "/")

function h_obj(x)
	return norm(x, 1)
end
function proxp(q, σ, xk, Δ)
	ProjB(wp) = min.(max.(wp, q .- σ), q .+ σ)
	ProjΔ(yp) = min.(max.(yp, -Δ), Δ)
	s = ProjΔ(ProjB(-xk))
	return s
end
ϕ.nonsmooth = h_obj 
ϕ.count = 0
ϕ.hist = []
g.func = h_obj
g.count = 0 
g.proxh = (z, α) -> sign.(z) .* max.(abs.(z) .- (λ * α) * ones(size(z)), zeros(size(z)))
g.λ = λ

parameters.f_obj = f_obj
parameters.h_obj = h_obj
parameters.λ = λ
parameters.χk = (s) -> norm(s, Inf)
parameters.ψχprox = proxp
l1binfv, l1binfp = evalwrapper(x0, xi, A, f_obj, h_obj,ϕ, g, λ, parameters, options, solverp,solverz, folder)




@info "running LS bfgs, h=l1, tr = l2"
xi = zeros(size(x0))
folder = string("figs/bpdn/LS_l1_B2/", compound, "/")
function proxp(q, σ, xk, Δ) # q = s - ν*g, ν*λ, xk, Δ - > basically inputs the value you need

	ProjB(y) = min.(max.(y, q .- σ), q .+ σ)
	froot(η) = η - norm(ProjB((-xk) .* (η / Δ)))

	# %do the 2 norm projection
	y1 = ProjB(-xk) # start with eta = tau

	if (norm(y1) <= Δ)
		y = y1  # easy case
	else
		η = fzero(froot, 1e-10, Inf)
		y = ProjB((-xk) .* (η / Δ))
	end
	if (norm(y) <= Δ)
		snew = y
	else
		snew = Δ .* y ./ norm(y)
	end
	return snew
end 

parameters.χk = (s) -> norm(s, 2)
parameters.ψχprox = proxp
ϕ.count = 0
ϕ.hist = []
g.count = 0
l1b2v, l1b2p = evalwrapper(x0, xi, A, f_obj, h_obj,ϕ, g, λ, parameters, options, solverp,solverz, folder)





@info "running LS bfgs, h=l0, tr = linf"
xi = zeros(size(x0))
folder = string("figs/bpdn/LS_l0_Binf/", compound, "/")

function h_obj(x)
	return norm(x, 0)
end
function proxp(q, σ, xk, Δ)
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

function proxl0(z, α)
    y = zeros(size(z))
    for i = 1:length(z)
        if abs(z[i]) > sqrt(2 * α * λ)
            y[i] = z[i]
        end
    end
    return y
end
parameters.h_obj = h_obj
parameters.χk = (s) -> norm(s, Inf)
parameters.ψχprox = proxp
ϕ.nonsmooth = h_obj 
ϕ.count = 0
ϕ.hist = []
g.count = 0 
g.func = h_obj
g.proxh = proxl0
l0binfv, l0binfp = evalwrapper(x0, xi, A, f_obj, h_obj,ϕ, g, λ, parameters, options, solverp,solverz, folder)






@info "running LS bfgs, h=B0, tr = linf"
λ = k
xi = zeros(size(x0))
folder = string("figs/bpdn/LS_B0_Binf/", compound, "/")

function h_obj(x)
	if norm(x, 0) ≤ λ
		h = 0
	else
		h = Inf
	end
	return h 
end

function proxp(q, σ, xk, Δ)
	ProjB(w) = min.(max.(w, -Δ), +Δ)
	w = q + xk 
	# find largest entries
	p = sortperm(abs.(w), rev=true)
	w[p[λ + 1:end]] .= 0 # set smallest to zero 
	s = ProjB(w - xk)# put all entries in projection?
	return s 
end
function proxb0(q, σ)
    # find largest entries
    p = sortperm(abs.(q), rev=true)
    q[p[λ + 1:end]] .= 0 # set smallest to zero 
    return q 
end

parameters.h_obj = h_obj
parameters.χk = (s) -> norm(s, Inf)
parameters.ψχprox = proxp
ϕ.nonsmooth = h_obj 
ϕ.count = 0
ϕ.hist = []
g.λ = λ
g.count = 0
g.func = h_obj
g.proxh = proxb0

b0binfv, b0binfp = evalwrapper(x0, xi, A, f_obj, h_obj,ϕ, g, λ, parameters, options, solverp,solverz, folder)

toplabs = ["\\(h=\\|\\cdot\\|_1\\), \\(\\Delta\\mathbb{B}_2\\)", "\\(h=\\|\\cdot\\|_0\\), \\(\\Delta\\mathbb{B}_\\infty\\)","\\(h=\\chi(\\cdot; \\lambda \\mathbb{B}_0)\\), \\(\\Delta\\mathbb{B}_\\infty\\)"]
xlabs = ["True", "TR", "PANOC", "ZFP", "TR", "PANOC", "ZFP", "TR", "PANOC", "ZFP"]

# pars = [l1b2p, l0binfp, b0binfp]
vals = hcat(l1b2v, l0binfv[:,2:end], b0binfv[:,2:end])

df = show_table(toplabs, vals, xlabs)
_ = write_table(toplabs, df, string("figs/bpdn/", "bpdn-table"))
