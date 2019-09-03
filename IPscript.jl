# Julia Testing function
# Generate Compressive Sensing Data
using Plots, Printf, Random, LinearAlgebra
include("DescentMethods.jl")
include("ProxProj.jl")
include("IP_alg.jl")
include("minconf_spg/SLIM_optim.jl")
using .SLIM_optim

#Here we just try to solve the Lasso Problem
#######
# min_x 1/2||Ax - b||^2


m,n = 100,100; # this is a under determine system
A = rand(m,n);
x0  = rand(n,);
b0 = A*x0;
b = b0 + 0.5*rand(m,);
cutoff = 0.0;
l = zeros(n,)+cutoff*ones(n,);
u = ones(n,)+cutoff*ones(n,); 


#set all options
minconf_options = spg_options(;optTol=1.0e-8, progTol=1.0e-10, verbose=0, feasibleInit=true, curvilinear=true, bbType=true, memory=1)
options = IP_options(A=A, b=b, minconf_options=minconf_options, l=l, u=u)


function LScustom(x)
    f = .5*norm(A*x-b)^2;
    g = A'*(A*x - b);
    h = A'*A; 
    return f, g, h
end

xin = (l+u)/2;

zl = ones(n,);
zu = ones(n,);

x, zjl, zju, j = IntPt_TR(xin, zl, zu, LScustom, options)


# plot(hispg[1:end-1], yaxis=:log, xlabel="Iteration", ylabel="Descent", title="Descent Comparison", label="ProxGrad")
# plot!(hisf[1:end-1], yaxis=:log, label="FISTA")
# savefig("temp.pdf")
