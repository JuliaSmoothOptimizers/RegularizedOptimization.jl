
#############################
# ======== IMPORTS ======== #
#############################
using Random, LinearAlgebra
using ProximalOperators, ProximalCore, ProximalAlgorithms
using ShiftedProximalOperators
using NLPModels, NLPModelsModifiers
using RegularizedOptimization, RegularizedProblems
using MLDatasets
using PrettyTables
using LaTeXStrings


# Local includes
include("comparison-config.jl")
using .ComparisonConfig: CFG, CFG2

include("Bench-utils.jl")
using .BenchUtils

#############################
# ===== Helper utils ====== #
#############################

# Generic config printer (works for both CFG / CFG2)
function print_config(cfg)
    println("Configuration:")
    for fld in fieldnames(typeof(cfg))
        val = getfield(cfg, fld)
        println(rpad("  $(String(fld))", 16), " = ", val)
    end
end

# Common QN selector
function ensure_qn(model, which::Symbol)
    which === :LBFGS && return LBFGSModel(model)
    which === :LSR1  && return LSR1Model(model)
    error("Unknown QN: $which (expected :LBFGS or :LSR1)")
end

# Close a PrettyTables Markdown string by adding a bottom rule
function close_markdown_table(table_str::AbstractString)
    lines = split(String(table_str), '\n')
    isempty(lines) && return table_str
    sep = length(lines) >= 2 ? lines[2] : repeat("-", 10)
    push!(lines, sep)
    return join(lines, '\n')
end

#############################
# ======= SVM bench ======= #
#############################

# Accuracy for SVM (as in original script)
acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100

function run_tr_svm!(model, x0; λ = 1.0, qn = :LSR1, atol = 1e-3, rtol = 1e-3, verbose = 0, sub_kwargs = (;))
    qn_model = ensure_qn(model, qn)
    reset!(qn_model)
    reg_nlp  = RegularizedNLPModel(qn_model, RootNormLhalf(λ))
    solver   = TRSolver(reg_nlp)
    stats    = RegularizedExecutionStats(reg_nlp)
    RegularizedOptimization.solve!(solver, reg_nlp, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    reset!(qn_model)  # Reset counters before timing
    reg_nlp  = RegularizedNLPModel(qn_model, RootNormLhalf(λ))
    solver   = TRSolver(reg_nlp)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nlp, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    return (
        name      = "TR ($(String(qn)), SVM)",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_obj(qn_model),
        gevals    = neval_grad(qn_model),
        proxcalls = get(stats.solver_specific, :prox_evals, missing),
        solution  = stats.solution,
        final_obj = obj(model, stats.solution)
    )
end

function run_r2n_svm!(model, x0; λ = 1.0, qn = :LBFGS, atol = 1e-3, rtol = 1e-3, verbose = 0, sub_kwargs = (;))
    qn_model = ensure_qn(model, qn)
    reset!(qn_model)
    reg_nlp  = RegularizedNLPModel(qn_model, RootNormLhalf(λ))
    solver   = R2NSolver(reg_nlp)
    stats    = RegularizedExecutionStats(reg_nlp)
    RegularizedOptimization.solve!(solver, reg_nlp, stats;
        x = x0, atol = atol, rtol = rtol,verbose = verbose, sub_kwargs = sub_kwargs)
    reset!(qn_model)  # Reset counters before timing
    reg_nlp = RegularizedNLPModel(qn_model, RootNormLhalf(λ)) # Re-create to reset prox eval count
    solver   = R2NSolver(reg_nlp)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nlp, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    return (
        name      = "R2N ($(String(qn)), SVM)",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_obj(qn_model),
        gevals    = neval_grad(qn_model),
        proxcalls = get(stats.solver_specific, :prox_evals, missing),
        solution  = stats.solution,
        final_obj = obj(model, stats.solution)
    )
end

function run_LM_svm!(nls_model, x0; λ = 1.0, atol = 1e-3, rtol = 1e-3, verbose = 0, sub_kwargs = (;))
    reg_nls  = RegularizedNLSModel(nls_model, RootNormLhalf(λ))
    solver   = LMSolver(reg_nls)
    stats    = RegularizedExecutionStats(reg_nls)
    RegularizedOptimization.solve!(solver, reg_nls, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    reset!(nls_model)  # Reset counters before timing
    reg_nls  = RegularizedNLSModel(nls_model, RootNormLhalf(λ))
    solver   = LMSolver(reg_nls)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nls, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    return (
        name      = "LM (SVM)",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_residual(nls_model),
        gevals    = neval_jtprod_residual(nls_model) + neval_jprod_residual(nls_model),
        proxcalls = get(stats.solver_specific, :prox_evals, missing),
        solution  = stats.solution,
        final_obj = obj(nls_model, stats.solution)
    )
end

function run_LMTR_svm!(nls_model, x0; λ = 1.0, atol = 1e-3, rtol = 1e-3, verbose = 0, sub_kwargs = (;))
    reg_nls  = RegularizedNLSModel(nls_model, RootNormLhalf(λ))
    solver   = LMTRSolver(reg_nls)
    stats    = RegularizedExecutionStats(reg_nls)
    RegularizedOptimization.solve!(solver, reg_nls, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    reset!(nls_model)  # Reset counters before timing
    reg_nls  = RegularizedNLSModel(nls_model, RootNormLhalf(λ))
    solver   = LMTRSolver(reg_nls)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nls, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    return (
        name      = "LMTR (SVM)",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_residual(nls_model),
        gevals    = neval_jtprod_residual(nls_model) + neval_jprod_residual(nls_model),
        proxcalls = get(stats.solver_specific, :prox_evals, missing),
        solution  = stats.solution,
        final_obj = obj(nls_model, stats.solution)
    )
end

function bench_svm!(cfg = CFG)
    Random.seed!(cfg.SEED)
    model, nls_train, _ = RegularizedProblems.svm_train_model()
    x0 = model.meta.x0

    results = NamedTuple[]
    (:TR    in cfg.RUN_SOLVERS) && push!(results, run_tr_svm!(model, x0; λ = cfg.LAMBDA_L0, qn = cfg.QN_FOR_TR, atol = cfg.TOL, rtol = cfg.RTOL, verbose = cfg.VERBOSE_RO, sub_kwargs = cfg.SUB_KWARGS_R2N))
    (:R2N   in cfg.RUN_SOLVERS) && push!(results, run_r2n_svm!(model, x0; λ = cfg.LAMBDA_L0, qn = cfg.QN_FOR_R2N, atol = cfg.TOL, rtol = cfg.RTOL, verbose = cfg.VERBOSE_RO, sub_kwargs = cfg.SUB_KWARGS_R2N))
    (:LM    in cfg.RUN_SOLVERS) && push!(results, run_LM_svm!(nls_train, x0; λ = cfg.LAMBDA_L0, atol = cfg.TOL, rtol = cfg.RTOL, verbose = cfg.VERBOSE_RO, sub_kwargs = cfg.SUB_KWARGS_R2N))
    (:LMTR  in cfg.RUN_SOLVERS) && push!(results, run_LMTR_svm!(nls_train, x0; λ = cfg.LAMBDA_L0, atol = cfg.TOL, rtol = cfg.RTOL, verbose = cfg.VERBOSE_RO, sub_kwargs = cfg.SUB_KWARGS_R2N))

    # Print quick summary
    println("\n=== SVM: solver comparison ===")
    for m in results
        println("\n→ ", m.name)
        println("  status         = ", m.status)
        println("  time (s)       = ", round(m.time, digits = 4))
        m.iters !== missing && println("  outer iters    = ", m.iters)
        println("  # f eval       = ", m.fevals)
        println("  # ∇f eval      = ", m.gevals)
        m.proxcalls !== missing && println("  # prox calls   = ", Int(m.proxcalls))
        println("  final objective= ", round(obj(model, m.solution), digits = 4))
        println("  accuracy (%)   = ", round(acc(residual(nls_train, m.solution)), digits = 1))
    end

    println("\nSVM Config:"); print_config(cfg)

    data_svm = [
        (; name=m.name,
           status=string(m.status),
           time=round(m.time, digits=4),
           fe=m.fevals,
           ge=m.gevals,
           prox = m.proxcalls === missing ? missing : Int(m.proxcalls),
           obj = round(obj(model, m.solution), digits=4))
        for m in results
    ]

    return data_svm
end

#############################
# ======= NNMF bench ====== #
#############################

function run_tr_nnmf!(model, x0; λ = 1.0, qn = :LSR1, atol = 1e-3, rtol = 1e-3, verbose = 0, sub_kwargs = (;), selected = nothing)
    qn_model = ensure_qn(model, qn)
    reset!(qn_model)
    reg_nlp  = RegularizedNLPModel(qn_model, NormL0(λ), selected)
    solver   = TRSolver(reg_nlp)
    stats    = RegularizedExecutionStats(reg_nlp)
    RegularizedOptimization.solve!(solver, reg_nlp, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    reset!(qn_model)  # Reset counters before timing
    reg_nlp  = RegularizedNLPModel(qn_model, NormL0(λ), selected) # Re-create to reset prox eval count
    solver   = TRSolver(reg_nlp)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nlp, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    return (
        name      = "TR ($(String(qn)), NNMF)",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_obj(qn_model),
        gevals    = neval_grad(qn_model),
        proxcalls = get(stats.solver_specific, :prox_evals, missing),
        solution  = stats.solution,
        final_obj = obj(model, stats.solution)
    )
end

function run_r2n_nnmf!(model, x0; λ = 1.0, qn = :LBFGS, atol = 1e-3, rtol = 1e-3, verbose = 0, sub_kwargs = (;), σk = 1e5, selected = nothing)
    qn_model = ensure_qn(model, qn)
    reset!(qn_model)
    reg_nlp  = RegularizedNLPModel(qn_model, NormL0(λ), selected)
    solver   = R2NSolver(reg_nlp)
    stats    = RegularizedExecutionStats(reg_nlp)
    RegularizedOptimization.solve!(solver, reg_nlp, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose,
        sub_kwargs = sub_kwargs)

    reset!(qn_model)  # Reset counters before timing
    reg_nlp  = RegularizedNLPModel(qn_model, NormL0(λ), selected) # Re-create to reset prox eval count
    solver   = R2NSolver(reg_nlp)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nlp, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose,
        sub_kwargs = sub_kwargs)
    return (
        name      = "R2N ($(String(qn)), NNMF)",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_obj(qn_model),
        gevals    = neval_grad(qn_model),
        proxcalls = get(stats.solver_specific, :prox_evals, missing),
        solution  = stats.solution,
        final_obj = obj(model, stats.solution)
    )
end

function run_LM_nnmf!(nls_model, x0; λ = 1.0, atol = 1e-3, rtol = 1e-3, verbose = 0, selected = nothing, sub_kwargs = (;))
    reg_nls  = RegularizedNLSModel(nls_model, NormL0(λ), selected)
    solver   = LMSolver(reg_nls)
    stats    = RegularizedExecutionStats(reg_nls)
    RegularizedOptimization.solve!(solver, reg_nls, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    reset!(nls_model)  # Reset counters before timing
    reg_nls  = RegularizedNLSModel(nls_model, NormL0(λ), selected)
    solver   = LMSolver(reg_nls)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nls, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    return (
        name      = "LM (NNMF)",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_residual(nls_model),
        gevals    = neval_jtprod_residual(nls_model) + neval_jprod_residual(nls_model),
        proxcalls = get(stats.solver_specific, :prox_evals, missing),
        solution  = stats.solution,
        final_obj = obj(nls_model, stats.solution)
    )
end

function run_LMTR_nnmf!(nls_model, x0; λ = 1.0, atol = 1e-3, rtol = 1e-3, verbose = 0, selected = nothing, sub_kwargs = (;))
    reg_nls  = RegularizedNLSModel(nls_model, NormL0(λ), selected)
    solver   = LMTRSolver(reg_nls)
    stats    = RegularizedExecutionStats(reg_nls)
    RegularizedOptimization.solve!(solver, reg_nls, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    reset!(nls_model)  # Reset counters before timing
    reg_nls  = RegularizedNLSModel(nls_model, NormL0(λ), selected)
    solver   = LMTRSolver(reg_nls)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nls, stats;
        x = x0, atol = atol, rtol = rtol, verbose = verbose, sub_kwargs = sub_kwargs)
    return (
        name      = "LMTR (NNMF)",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_residual(nls_model),
        gevals    = neval_jtprod_residual(nls_model) + neval_jprod_residual(nls_model),
        proxcalls = get(stats.solver_specific, :prox_evals, missing),
        solution  = stats.solution,
        final_obj = obj(nls_model, stats.solution)
    )
end

function bench_nnmf!(cfg = CFG2; m = 100, n = 50, k = 5)
    Random.seed!(cfg.SEED)

    model, nls_model, _, selected = nnmf_model(m, n, k)

    # build x0 on positive orthant as original
    x0 = max.(rand(model.meta.nvar), 0.0)

    # heuristic lambda (copied logic)
    cfg.LAMBDA_L0 = norm(grad(model, rand(model.meta.nvar)), Inf) / 200

    results = NamedTuple[]
    (:TR  in cfg.RUN_SOLVERS) && push!(results, run_tr_nnmf!(model, x0; λ = cfg.LAMBDA_L0, qn = cfg.QN_FOR_TR, atol = cfg.TOL, rtol = cfg.RTOL, verbose = cfg.VERBOSE_RO, sub_kwargs = cfg.SUB_KWARGS_R2N, selected = selected))
    (:R2N in cfg.RUN_SOLVERS) && push!(results, run_r2n_nnmf!(model, x0; λ = cfg.LAMBDA_L0, qn = cfg.QN_FOR_R2N, atol = cfg.TOL, rtol = cfg.RTOL, verbose = cfg.VERBOSE_RO, sub_kwargs = cfg.SUB_KWARGS_R2N, selected = selected))
    (:LM  in cfg.RUN_SOLVERS) && push!(results, run_LM_nnmf!(nls_model, x0; λ = cfg.LAMBDA_L0, atol = cfg.TOL, rtol = cfg.RTOL, verbose = cfg.VERBOSE_RO, selected = selected, sub_kwargs = cfg.SUB_KWARGS_R2N))
    (:LMTR in cfg.RUN_SOLVERS) && push!(results, run_LMTR_nnmf!(nls_model, x0; λ = cfg.LAMBDA_L0, atol = cfg.TOL, rtol = cfg.RTOL, verbose = cfg.VERBOSE_RO, selected = selected, sub_kwargs = cfg.SUB_KWARGS_R2N))

    println("\n=== NNMF: solver comparison ===")
    for m in results
        println("\n→ ", m.name)
        println("  status         = ", m.status)
        println("  time (s)       = ", round(m.time, digits = 4))
        m.iters !== missing && println("  outer iters    = ", m.iters)
        println("  # f eval       = ", m.fevals)
        println("  # ∇f eval      = ", m.gevals)
        m.proxcalls !== missing && println("  # prox calls   = ", Int(m.proxcalls))
        println("  final objective= ", round(obj(model, m.solution), digits = 4))
    end

    println("\nNNMF Config:"); print_config(cfg)

    data_nnmf = [
        (; name=m.name,
           status=string(m.status),
           time=round(m.time, digits=4),
           fe=m.fevals,
           ge=m.gevals,
           prox = m.proxcalls === missing ? missing : Int(m.proxcalls),
           obj = round(m.final_obj, digits=4))
        for m in results
    ]

    return data_nnmf
end

# #############################
# # ========= Main ========== #
# #############################

function main(latex_out = false)
    data_svm  = bench_svm!(CFG)
    data_nnmf = bench_nnmf!(CFG2)

    all_data = vcat(data_svm, data_nnmf)

    println("\n=== Full Benchmark Table ===")
    # what is inside the table
    for row in all_data
        println(row)
    end

    # save as latex format
    if latex_out

        table_str = pretty_table(String, all_data;
                header = ["Method", "Status", L"$t$($s$)", L"$\#f$", L"$\#\nabla f$", L"$\#prox$", "Objective"],
                backend = Val(:latex),
                alignment = [:l, :c, :r, :r, :r, :r, :r],
            )

        open("Benchmark.tex", "w") do io
            write(io, table_str)
        end
    end
    return nothing
end

