#===========================================================================
	Proximal Operator of General lp norm
===========================================================================#

function e_lp_scalar(x, z, a, p)
	@assert(p ≥ 0.0);
	@assert(a > 0.0);
	@assert(size(x) == size(z));
	@assert(ndims(z) == 0);

	T = typeof(x);
	val = 0.5*(x - z)^2/a;
	p == 0.0 ? val += T(iszero(x)) : val += abs(x)^p;
	return val
end

function e_lp(x, z, a, p)
	@assert(p ≥ 0.0);
	@assert(a > 0.0);
	@assert(size(x) == size(z));

	if ndims(x) == 0
		println("scalar detected, calling e_lp_scalar instead.");
		return e_lp_scalar(x, z, a::Float64, p)
	else
		T = eltype(x);
		n = length(x);
		f(x) = abs(x)^p;
		p == 0.0 && (f(x) == T(iszero(x)));
		val = 0.0;
		for i = 1:n
			val += 0.5*(x[i] - z[i])^2/a + f(x[i]);
		end
		return val
	end
end

function prox_lp_scalar(z, a, p)
	@assert(p ≥ 0.0);
	@assert(a ≥ 0.0);
	@assert(ndims(z) == 0);

	x = z;
	p ≡ 0.0 ? x = prox_l0_scalar(z, a, p) :
	p < 1.0 ? x = prox_ll_scalar(z, a, p) :
	p ≡ 1.0 ? x = prox_l1_scalar(z, a, p) : x = prox_lr_scalar(z, a::Float64, p);

	return x
end

function prox_lp(z, a, p)
	x = 0.0*z
	@assert(p ≥ 0.0);
	# @assert(a ≥ 0.0);
	@assert(size(x) == size(z));
	if ndims(x) == 0
		println("scalar detected, calling prox_lp_scalar instead.");
		return prox_lp_scalar(z, a, p)
	else
		p ≡ 0.0 ? x=prox_l0(z, a, p) :
		p < 1.0 ? x=prox_ll(z, a, p) :
		p ≡ 1.0 ? x=prox_l1(z, a, p) : x=prox_lr(z, a::Float64, p);
	end
	return x
end


function prox_l0_scalar(z, a, p)
	κ = sqrt(2.0*a);
	x = z;
	abs(z) ≤ κ && (x = 0.0);
	return x
end

function prox_ll_scalar(z, a, p; itm=10, tol=1e-6)
	ρ = abs(z);
	s = sign(z);

	x̃ = (a*p*(1.0-p))^(1.0/(2.0-p));
	g = (x̃ - ρ)/a + p*x̃^(p-1.0);

	g ≥ 0.0 && (return 0.0);

	x̄ = ρ;
	g = (x̄ - ρ)/a + p*x̄^(p-1.0);
	h = 1.0/a + p*(p-1.0)*x̄^(p-2.0);

	noi = 0;
	while abs(g) ≥ tol
		d = g/h;
		if x̄ < d + x̃
			x̄ = 0.1*x̄ + 0.9*x̃;
		else
			x̄ = x̄ - d;
		end
		g = (x̄ - ρ)/a + p*x̄^(p-1.0);
		h = 1.0/a + p*(p-1.0)*x̄^(p-2.0);
		noi = noi + 1;
		noi ≥ itm && break;
	end
	fx̄ = 0.5*(x̄ - ρ)^2/a + x̄^p;
	f0 = 0.5*ρ^2/a;

	x  = s;
	f0 ≤ fx̄ ? x = 0.0 : x *= x̄;

	# @show noi;
	return x
end

function prox_l1_scalar(z, a, ρ)
	ρ = abs(z);
	s = sign(z);
	x = s*max(0.0, ρ - a);
	return x
end

function prox_lr_scalar(z, a, ρ; itm=10, tol=1e-6)
	ρ = abs(z);
	s = sign(z);

	x = ρ/(1.0/a + 2.0);
	g = (x - ρ)/a + p*x^(p-1.0);
	h = 1.0/a + p*(p-1.0)*x^(p-2.0);

	noi = 0;
	while abs(g) ≥ tol
		d = g/h;
		if x < d
			x *= 0.1;
		else
			x = x - d;
		end
		g = (x - ρ)/a + p*x^(p-1.0);
		h = 1.0/a + p*(p-1.0)*x^(p-2.0);
		noi = noi + 1;
		noi ≥ itm && break;
	end
	x *= s;

	# @show noi;
	return x
end

##### scalar a #####
#note all of these have been changed to produce an output variable instead of modifying in place
function prox_l0(z, a::Float64, p)
	x = 0.0*z
	for i in eachindex(x)
		x[i] = prox_l0_scalar(z[i], a, p);
	end
	return x
end

function prox_ll(z, a::Float64, p; itm=10, tol=1e-6)
	x = 0.0*z
	for i in eachindex(x)
		x[i] = prox_ll_scalar(z[i], a, p, itm=itm, tol=tol);
	end
	return x
end

function prox_l1(z, a::Float64, p)
	x = 0.0*z
	for i in eachindex(x)
		x[i] = prox_l1_scalar(z[i], a, p);
	end
	return x
end

function prox_lr(z, a::Float64, p; itm=10, tol=1e-6)
	x = 0.0*z
	for i in eachindex(x)
		x[i] = prox_l0_scalar(z[i], a, p, itm=itm, tol=tol);
	end
	return x
end

##### vector a #####
function prox_l0(z, a::Vector{Float64}, p)
	x = 0.0*z
	for i in eachindex(x)
		x[i] = prox_l0_scalar(z[i], a[i], p);
	end
	return x
end

function prox_ll(z, a::Vector{Float64}, p; itm=10, tol=1e-6)
	x = 0.0*z
	for i in eachindex(x)
		x[i] = prox_ll_scalar(z[i], a[i], p, itm=itm, tol=tol);
	end
	return x
end

function prox_l1(z, a::Vector{Float64}, p)
	x = 0.0*z
	for i in eachindex(x)
		x[i] = prox_l1_scalar(z[i], a[i], p);
	end
	return x
end

function prox_lr(z, a::Vector{Float64}, p; itm=10, tol=1e-6)
	x = 0.0*z
	for i in eachindex(x)
		x[i] = prox_l0_scalar(z[i], a[i], p, itm=itm, tol=tol);
	end
	return x
end
