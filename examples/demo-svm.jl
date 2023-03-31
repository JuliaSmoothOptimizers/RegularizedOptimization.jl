using Random
using LinearAlgebra
using ProximalOperators, ShiftedProximalOperators, MLDatasets, RegularizedProblems
using NLPModels, NLPModelsModifiers #ReverseADNLSModels
using RegularizedOptimization
using DataFrames
import SolverBenchmark

include("plot-utils-svm.jl")

Random.seed!(1234)

function demo_solver(nlp_tr, nls_tr, sol_tr, nlp_test, nls_test, sol_test, h, χ, suffix="l0-linf")
    options = ROSolverOptions(ν=1.0, β=1e16, ϵa=1e-4, ϵr=1e-4, verbose=10, σmin=1e-5)
    suboptions = ROSolverOptions(maxIter = 100)
    acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100

    @info "using R2 to solve with" h
    reset!(nlp_tr)
    R2_out = R2(nlp_tr, h, options, x0=nlp_tr.meta.x0)
    nr2 = neval_obj(nlp_tr)
    ngr2 = neval_grad(nlp_tr)
    r2train = residual(nls_tr, R2_out.solution) #||e - tanh(b * <A, x>)||^2, b ∈ {-1,1}^n
    r2test = residual(nls_test, R2_out.solution)
    @show acc(r2train), acc(r2test)
    r2dec = plot_svm(R2_out, R2_out.solution, "r2-$(suffix)")

    @info "using TR to solve with" h χ
    reset!(nlp_tr)
    TR_out = TR(nlp_tr, h, χ, options, x0=nlp_tr.meta.x0, subsolver_options = suboptions)
    trtrain = residual(nls_tr, TR_out.solution)
    trtest = residual(nls_test, TR_out.solution)
    ntr = neval_obj(nlp_tr)
    ngtr = neval_grad(nlp_tr)
    @show acc(trtrain), acc(trtest)
    trdec = plot_svm(TR_out, TR_out.solution, "tr-$(suffix)")

    @info " using LMTR to solve with" h χ
    reset!(nls_tr)
    LMTR_out = LMTR(nls_tr, h, χ, options, x0=nls_tr.meta.x0, subsolver_options = suboptions)
    lmtrtrain = residual(nls_tr, LMTR_out.solution)
    lmtrtest = residual(nls_test, LMTR_out.solution)
    nlmtr = neval_residual(nls_tr)
    nglmtr = neval_jtprod_residual(nls_tr) + neval_jprod_residual(nls_tr)
    @show acc(lmtrtrain), acc(lmtrtest)
    lmtrdec = plot_svm(LMTR_out, LMTR_out.solution, "lmtr-$(suffix)")

    @info " using LMTR to solve with" h χ
    reset!(nls_tr)
    LM_out = LM(nls_tr, h, options, x0=nls_tr.meta.x0, subsolver_options = suboptions)
    lmtrain = residual(nls_tr, LM_out.solution)
    lmtest = residual(nls_test, LM_out.solution)
    nlm = neval_residual(nls_tr)
    nglm = neval_jtprod_residual(nls_tr) + neval_jprod_residual(nls_tr)
    @show acc(lmtrain), acc(lmtest)
    lmdec = plot_svm(LM_out, LM_out.solution, "lm-$(suffix)")

    # c = PGFPlots.Axis(
    #     [
    #         PGFPlots.Plots.Linear(1:length(r2dec), r2dec, mark="none", style="black, dotted", legendentry="R2"),
    #         PGFPlots.Plots.Linear(1:length(trdec), trdec, mark="none", style="black, dashed", legendentry="TR"),
    #         PGFPlots.Plots.Linear(LM_out.solver_specific[:ResidHist], lmdec, mark="none", style="black, thick", legendentry="LM"),
    #         PGFPlots.Plots.Linear(LMTR_out.solver_specific[:ResidHist], lmtrdec, mark="none", style = "black, very thin", legendentry="LMTR"),
    #     ],
    #     xlabel="\$ k^{th}\$   \$ f \$ Eval",
    #     ylabel="Objective Value",
    #     ymode="log",
    #     xmode="log",
    # )
    # PGFPlots.save("svm-objdec.tikz", c, include_preamble=false)

    # temp = hcat([R2_out.solver_specific[:Fhist][end], R2_out.solver_specific[:Hhist][end],R2_out.objective, acc(r2train), acc(r2test), nr2, ngr2, sum(R2_out.solver_specific[:SubsolverCounter]), R2_out.elapsed_time],
    #     [TR_out.solver_specific[:Fhist][end], TR_out.solver_specific[:Hhist][end], TR_out.objective, acc(trtrain), acc(trtest), ntr, ngtr, sum(TR_out.solver_specific[:SubsolverCounter]), TR_out.elapsed_time],
    #     [LM_out.solver_specific[:Fhist][end], LM_out.solver_specific[:Hhist][end], LM_out.objective, acc(lmtrain), acc(lmtest), nlm, nglm, sum(LM_out.solver_specific[:SubsolverCounter]), LM_out.elapsed_time],
    #     [LMTR_out.solver_specific[:Fhist][end], LMTR_out.solver_specific[:Hhist][end], LMTR_out.objective, acc(lmtrtrain), acc(lmtrtest), nlmtr, nglmtr, sum(LMTR_out.solver_specific[:SubsolverCounter]), LMTR_out.elapsed_time])'

    # df = DataFrame(temp, [:f, :h, :fh, :x,:xt, :n, :g, :p, :s])
    # T = []
    # for i = 1:nrow(df)
    #   push!(T, Tuple(df[i, [:x, :xt]]))
    # end
    # select!(df, Not(:xt))
    # df[!, :x] = T
    # df[!, :Alg] = ["R2", "TR", "LM", "LMTR"]
    # select!(df, :Alg, Not(:Alg), :)
    # fmt_override = Dict(:Alg => "%s",
    #     :f => "%10.2f",
    #     :h => "%10.2f",
    #     :fh => "%10.2f",
    #     :x => "%10.2f, %10.2f",
    #     :n => "%i",
    #     :g => "%i",
    #     :p => "%i",
    #     :s => "%02.2f")
    # hdr_override = Dict(:Alg => "Alg",
    #     :f => "\$ f \$",
    #     :h => "\$ h \$",
    #     :fh => "\$ f+h \$",
    #     :x => "(Train, Test)",
    #     :n => "\\# \$f\$",
    #     :g => "\\# \$ \\nabla f \$",
    #     :p => "\\# \$ \\prox{}\$",
    #     :s => "\$t \$ (s)")
    # open("svm.tex", "w") do io
    #     SolverBenchmark.pretty_latex_stats(io, df,
    #         col_formatters=fmt_override,
    #         hdr_override=hdr_override)
    # end
end

function demo_svm()
    ## load phishing data from libsvm
    # A = readdlm("data_matrix.txt")
    # b = readdlm("label_vector.txt")

    # # sort into test/trainig
    # test_ind = randperm(length(b))[1:Int(floor(length(b)*.1))]
    # train_ind = setdiff(1:length(b), test_ind)
    # btest = b[test_ind]
    # Atest = A[test_ind,:]'
    # btrain = b[train_ind]
    # Atrain = A[train_ind,:]'

    nlp_train, nls_train, sol_train = svm_train_model()#Atrain, btrain) #
    nlp_test, nls_test, sol_test = svm_test_model()#Atest, btest)
    nlp_train = LSR1Model(nlp_train)
    λ = 1e-1
    h = RootNormLhalf(λ)
    χ = NormLinf(1.0)


    demo_solver(nlp_train, nls_train, sol_train, nlp_test, nls_test, sol_test, h, χ, "lhalf-linf")
end

demo_svm()

