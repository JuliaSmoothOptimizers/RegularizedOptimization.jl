using Random
using LinearAlgebra
using ProximalOperators, ShiftedProximalOperators, RegularizedProblems
using NLPModels, NLPModelsModifiers
using RegularizedOptimization
using MLDatasets

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
end

function demo_svm()
    nlp_train, nls_train, sol_train = RegularizedProblems.svm_train_model()
    nlp_test, nls_test, sol_test = RegularizedProblems.svm_test_model()
    nlp_train = LSR1Model(nlp_train)
    λ = 1e-1
    h = RootNormLhalf(λ)
    χ = NormLinf(1.0)

    demo_solver(nlp_train, nls_train, sol_train, nlp_test, nls_test, sol_test, h, χ, "lhalf-linf")
end

demo_svm()

