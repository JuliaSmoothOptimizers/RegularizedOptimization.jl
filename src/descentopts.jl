export s_options

mutable struct s_params
	optTol
	maxIter
	verbose
	restart
	β
	α
	λ
	η
    η_factor
    Δ
    WoptTol
    ∇fk
    Bk
    xk

end


function s_options(β;optTol=1f-6, maxIter=10000, verbose=0, restart=10, λ=1.0, η =1.0, η_factor=.9, Δ=1.0, α = .95,
     WoptTol=1f-6, ∇fk = Vector{Float64}(undef,0), Bk = Array{Float64}(undef, 0,0), xk=Vector{Float64}(undef,0))

	return s_params(optTol, maxIter, verbose, restart, β,α, λ, η, η_factor,Δ, WoptTol, ∇fk, Bk,xk)

end
