
#############################
# ======== IMPORTS ======== #
#############################
using Random, LinearAlgebra
using ProximalOperators, ProximalCore, ProximalAlgorithms
using ADNLPModels, NLPModels, NLPModelsModifiers
using RegularizedOptimization, RegularizedProblems
using ShiftedProximalOperators
using MLDatasets

include("comparison-config.jl")
using .ComparisonConfig: CFG3

include("Bench-utils.jl")
using .BenchUtils

function print_config(CFG3)
    println("Configuration:")
    println("  SEED            = $(CFG3.SEED)")
    println("  LAMBDA_L0       = $(CFG3.LAMBDA_L0)")
    println("  TOL             = $(CFG3.TOL)")
    println("  RTOL            = $(CFG3.RTOL)")
    println("  MAXIT_PANOC     = $(CFG3.MAXIT_PANOC)")
    println("  VERBOSE_PANOC   = $(CFG3.VERBOSE_PANOC)")
    println("  VERBOSE_RO      = $(CFG3.VERBOSE_RO)")
    println("  RUN_SOLVERS     = $(CFG3.RUN_SOLVERS)")
    println("  QN_FOR_TR       = $(CFG3.QN_FOR_TR)")
    println("  QN_FOR_R2N      = $(CFG3.QN_FOR_R2N)")
    println("  SUB_KWARGS_R2N  = $(CFG3.SUB_KWARGS_R2N)")
    println("  SIGMAK_R2N      = $(CFG3.SIGMAK_R2N)")
    println("  X0_SCALAR       = $(CFG3.X0_SCALAR)")
    println("  PRINT_TABLE     = $(CFG3.PRINT_TABLE)")   
    println("  OPNORM_MAXITER  = $(CFG3.OPNORM_MAXITER)")
    println("  HESSIAN_SCALE  = $(CFG3.HESSIAN_SCALE)")
end

acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100 # for SVM

#############################
# ===== PROBLÈME (NNMF) ===== #
#############################
Random.seed!(CFG3.SEED)

m, n, k = 100, 50, 5
model, nls_model, A, selected = nnmf_model(m, n, k)

x0 = rand(model.meta.nvar)

CFG3.LAMBDA_L0 = norm(grad(model, rand(model.meta.nvar)), Inf) / 200
#############################
# ======= PANOC run ======= #
#############################
function run_panoc!(model, x0; λ = 1.0, maxit = 500, tol = 1e-3, verbose = false)
    λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 200
    f = BenchUtils.Counting(model)
    g = BenchUtils.Counting(NormL0(λ))
    algo = ProximalAlgorithms.PANOC(maxit = maxit, tol = tol, verbose = verbose)
    t = @elapsed x̂, it = algo(x0 = x0, f = f, g = g)
    metrics = (
        name      = "PANOC",
        status    = "first_order",
        time      = t,
        iters     = it,
        fevals    = f.eval_count,
        gevals    = f.gradient_count,
        proxcalls = g.prox_count,
        solution  = x̂,
    )
    return metrics
end

#############################
# ======== TR run ========= #
#############################
function ensure_qn(model, which::Symbol)
    which === :LBFGS && return LBFGSModel(model)
    which === :LSR1  && return LSR1Model(model)
    error("QN inconnu: $which (attendu :LBFGS ou :LSR1)")
end

function run_tr!(model, x0; λ = 1.0, qn = :LSR1, atol = 1e-3, rtol = 1e-3, verbose = 0, sub_kwargs = (;), selected = selected, opnorm_maxiter = 20)
    qn_model = ensure_qn(model, qn)
    reset!(qn_model)  # reset des compteurs
    reg_nlp  = RegularizedNLPModel(qn_model, NormL0(λ), selected)
    solver   = TRSolver(reg_nlp)
    stats    = RegularizedExecutionStats(reg_nlp)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nlp, stats;
                                                x = x0, atol = atol, rtol = rtol, verbose = verbose, opnorm_maxiter = opnorm_maxiter, sub_kwargs = sub_kwargs)
    metrics = (
        name      = "TR($(String(qn)))",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_obj(qn_model),
        gevals    = neval_grad(qn_model),
        proxcalls = stats.solver_specific[:prox_evals],
        solution  = stats.solution,
    )
    return metrics
end

#############################
# ======== R2N run ======== #
#############################
function run_r2n!(model, x0; λ = 1.0, qn = :LBFGS, atol = 1e-3, rtol = 1e-3, verbose = 0, sub_kwargs = (;), σk = 1e5, opnorm_maxiter = 20)
    qn_model = ensure_qn(model, qn)
    reset!(qn_model)
    reg_nlp  = RegularizedNLPModel(qn_model, NormL0(λ))
    solver   = R2NSolver(reg_nlp, m_monotone = 10)
    stats    = RegularizedExecutionStats(reg_nlp)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nlp, stats;
                                                x = x0, atol = atol, rtol = rtol, σk = σk,
                                                verbose = verbose, sub_kwargs = sub_kwargs, opnorm_maxiter = opnorm_maxiter)
    metrics = (
        name      = "R2N($(String(qn))) Nonmonotone",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_obj(qn_model),
        gevals    = neval_grad(qn_model),
        proxcalls = stats.solver_specific[:prox_evals],
        solution  = stats.solution,
    )
    return metrics
end

#############################
# ======== LM run ======== #
#############################
function run_LM!(nls_model, x0; λ = 1.0, atol = 1e-3, rtol = 1e-3, verbose = 0, σk = 1e0)
    reg_nls  = RegularizedNLSModel(nls_model, NormL0(λ))
    solver   = LMSolver(reg_nls)
    stats    = RegularizedExecutionStats(reg_nls)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nls, stats;
                                                x = x0, atol = atol, rtol = rtol, σk = σk,
                                                verbose = verbose)
    metrics = (
        name      = "LM",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_residual(nls_model),
        gevals    = neval_jtprod_residual(nls_model) + neval_jprod_residual(nls_model),
        proxcalls = stats.solver_specific[:prox_evals],
        solution  = stats.solution,
    )
    return metrics
end

#############################
# ====== LANCEMENTS ======= #
#############################
results = NamedTuple[]

if :TR in CFG3.RUN_SOLVERS
    push!(results, run_tr!(model, x0; λ = CFG3.LAMBDA_L0, qn = CFG3.QN_FOR_TR, atol = CFG3.TOL, rtol = CFG3.RTOL, verbose = CFG3.VERBOSE_RO, sub_kwargs = CFG3.SUB_KWARGS_R2N, opnorm_maxiter = CFG3.OPNORM_MAXITER))
end
if :R2N in CFG3.RUN_SOLVERS
    push!(results, run_r2n!(model, x0; λ = CFG3.LAMBDA_L0, qn = CFG3.QN_FOR_R2N, atol = CFG3.TOL, rtol = CFG3.RTOL,
                            verbose = CFG3.VERBOSE_RO, sub_kwargs = CFG3.SUB_KWARGS_R2N, σk = CFG3.SIGMAK_R2N, opnorm_maxiter = CFG3.OPNORM_MAXITER))
end
if :LM in CFG3.RUN_SOLVERS
    push!(results, run_LM!(nls_model, x0; λ = CFG3.LAMBDA_L0, atol = CFG3.TOL, rtol = CFG3.RTOL,
                            verbose = CFG3.VERBOSE_RO, σk = CFG3.SIGMAK_R2N))
end

using PrettyTables

#############################
# ===== AFFICHAGE I/O ===== #
#############################


println("\n=== Comparaison solveurs ===")
for m in results
    println("\n→ ", m.name)
    println("  statut       = ", m.status)
    println("  temps (s)    = ", round(m.time, digits=4))
    if m.iters !== missing
        println("  itérations   = ", m.iters)
    end
    println("  # f eval     = ", m.fevals)
    println("  # ∇f eval    = ", m.gevals)
    if m.proxcalls !== missing
        println("  # prox appels = ", Int(m.proxcalls))
    end
    println("  objective final", " = ", round(obj(model, m.solution), digits=4))
end

println("\n")
print_config(CFG3)

if CFG3.PRINT_TABLE
    println("\nSummary :")
    # Construire les données pour la table
    data = [
    (; name=m.name,
       status=string(m.status),
       time=round(m.time, digits=4),
       fe=m.fevals,
       ge=m.gevals,
       prox = m.proxcalls === missing ? missing : Int(m.proxcalls))
    for m in results
]

    # En-têtes
    table_str = pretty_table(String,
           data;
           header = ["Method", "Status", "Time (s)", "#f", "#∇f", "#prox"],
           tf = tf_unicode,
           alignment = [:l, :c, :r, :r, :r, :r],
           crop = :none,
       )
    

    open("NNMF-comparison-f.txt", "w") do io
    write(io, table_str)
    end
end