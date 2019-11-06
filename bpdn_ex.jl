# Julia Testing function
# Generate Compressive Sensing Data
using Plots, Printf, Random, LinearAlgebra, Convex, SCS
include("DescentMethods.jl")
include("ProxProj.jl")
include("IP_alg.jl")
include("barrier.jl")
include("minconf_spg/SLIM_optim.jl")
include("minconf_spg/oneProjector.jl")
using .SLIM_optim

#Here we just try to solve the l2-norm Problem over the l1 trust region 
#######
# min_x 1/2||Ax - b||^2


m,n = 200,100; # this is a under determined system
# m, n = 10, 2
A = rand(m,n);
x0  = rand(n,);
b0 = A*x0;
b = b0 + 0.5*rand(m,);
cutoff = 0.0;
l = zeros(n,)+cutoff*ones(n,);
u = ones(n,)+cutoff*ones(n,); 





#define your objective function 
function LScustom(x)
    f = .5*norm(A*x-b)^2;
    g = A'*(A*x - b);
    h = A'*A; 
    return f, g, h
end

function proxG(x,λ,α)
    n = length(x)
    for i = 1:n
        x[i] > α*λ ? x[i] -= α*λ :
        x[i] <-α*λ ? x[i] += α*λ : x[i] = 0.0;
    end
    return x
    # return sign.(x).*max(abs.(x).-(α)*ones(size(x)), zeros(size(x)))
end
#set all options
#uncomment for OTHER test
first_order_options = s_options(norm(A)^(2.0) ;optTol=1.0e-3, verbose=0)
parameters = IP_struct(LScustom; l=l, u=u, tr_options = first_order_options, tr_projector_alg=FISTA, projector=proxG)
options = IP_options()
#put in your initial guesses 
x = (l+u)/2;
zl = ones(n,);
zu = ones(n,);

X = Variable(n)
problem = minimize(sumsquares(A * X - b), X>=l, X<=u)
solve!(problem, SCSSolver())




x, zl, zu = barrier_alg(x,zl, zu, parameters, options)


#print out l2 norm difference and plot the two x values 
@printf("l2-norm TR: %5.5e\n", norm(x - x0))
@printf("l2-norm CVX: %5.5e\n", norm(X.value - x0))
@printf("TR vs CVX relative error: %5.5e\n", norm(X.value - x)/norm(X.value))
plot(x0, xlabel="i^th index", ylabel="x", title="TR vs True x", label="True x")
plot!(x, label="tr", marker=2)
plot!(X.value, label="cvx")
savefig("xcomp.pdf")

plot(b0, xlabel="i^th index", ylabel="b", title="TR vs True x", label="True b")
plot!(b, label="Observed")
plot!(A*x, label="A*x: TR", marker=2)
plot!(A*X.value, label="A*x: CVX")
savefig("bcomp.pdf")