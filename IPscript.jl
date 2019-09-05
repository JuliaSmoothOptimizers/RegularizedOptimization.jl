# Julia Testing function
# Generate Compressive Sensing Data
using Plots, Printf, Random, LinearAlgebra
include("DescentMethods.jl")
include("ProxProj.jl")
include("IP_alg.jl")
include("minconf_spg/SLIM_optim.jl")
using .SLIM_optim

#Here we just try to solve the l2-norm Problem
#######
# min_x 1/2||Ax - b||^2


m,n = 100,100; # this is a under determined system
A = rand(m,n);
x0  = rand(n,);
b0 = A*x0;
b = b0 + 0.5*rand(m,);
cutoff = 0.0;
l = zeros(n,)+cutoff*ones(n,);
u = ones(n,)+cutoff*ones(n,); 


#set all options
minconf_options = spg_options(;optTol=1.0e-8, progTol=1.0e-10, verbose=0, feasibleInit=true, curvilinear=true, bbType=true, memory=1)
options = IP_options(minconf_options=minconf_options, l=l, u=u)

#define your objective function 
function LScustom(x)
    f = .5*norm(A*x-b)^2;
    g = A'*(A*x - b);
    h = A'*A; 
    return f, g, h
end

#put in your initial guesses 
xin = (l+u)/2;
zl = ones(n,);
zu = ones(n,);

#throw it into the trust region 
x, zjl, zju, j = IntPt_TR(xin, zl, zu, LScustom, options)

#print out l2 norm difference and plot the two x values 
@printf("l2-norm: %5.5e\n", norm(x - x0))
plot(x0, xlabel="i^th index", ylabel="x", title="TR vs True x", label="True x")
plot!(x, label="x_tr")
savefig("xcomp.pdf")

plot(b0, xlabel="i^th index", ylabel="b", title="TR vs True x", label="True b")
plot!(b, label="Observed")
plot!(A*x, label="A*x")
savefig("bcomp.pdf")