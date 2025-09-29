
#############################
# ======== IMPORTS ======== #
#############################
using Random, LinearAlgebra
using ProximalOperators, ProximalCore, ProximalAlgorithms
using ADNLPModels, NLPModels, NLPModelsModifiers
using ShiftedProximalOperators
using RegularizedOptimization, RegularizedProblems
using MLDatasets

include("comparison-config.jl")
using .ComparisonConfig: CFG2

include("Bench-utils.jl")
using .BenchUtils

function print_config(CFG2)
    println("Configuration:")
    println("  SEED            = $(CFG2.SEED)")
    println("  LAMBDA_L0       = $(CFG2.LAMBDA_L0)")
    println("  TOL             = $(CFG2.TOL)")
    println("  RTOL            = $(CFG2.RTOL)")
    println("  MAXIT_PANOC     = $(CFG2.MAXIT_PANOC)")
    println("  VERBOSE_PANOC   = $(CFG2.VERBOSE_PANOC)")
    println("  VERBOSE_RO      = $(CFG2.VERBOSE_RO)")
    println("  RUN_SOLVERS     = $(CFG2.RUN_SOLVERS)")
    println("  QN_FOR_TR       = $(CFG2.QN_FOR_TR)")
    println("  QN_FOR_R2N      = $(CFG2.QN_FOR_R2N)")
    println("  SUB_KWARGS_R2N  = $(CFG2.SUB_KWARGS_R2N)")
    println("  SIGMAK_R2N      = $(CFG2.SIGMAK_R2N)")
    println("  X0_SCALAR       = $(CFG2.X0_SCALAR)")
    println("  PRINT_TABLE     = $(CFG2.PRINT_TABLE)")   
    println("  OPNORM_MAXITER  = $(CFG2.OPNORM_MAXITER)")
    println("  HESSIAN_SCALE  = $(CFG2.HESSIAN_SCALE)")
    println("  M_MONOTONE     = $(CFG2.M_MONOTONE)")
end

acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100 # for SVM

#############################
# ===== PROBLÈME (SVM) ===== #
#############################
Random.seed!(CFG2.SEED)

model, nls_train, _ = RegularizedProblems.svm_train_model()
x0 = model.meta.x0

#############################
# ======= PANOC run ======= #
#############################
function run_panoc!(model, x0; λ = 1.0, maxit = 500, tol = 1e-3, verbose = false)
   # BenchUtils.make_adnlp_compatible!()
    f = BenchUtils.Counting(model)
    g = BenchUtils.Counting(RootNormLhalf(λ))
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

function run_tr!(model, x0; λ = 1.0, qn = :LSR1, atol = 1e-3, rtol = 1e-3, verbose = 0, sub_kwargs = (;), opnorm_maxiter = 20)
    qn_model = ensure_qn(model, qn)
    reset!(qn_model)  # reset des compteurs
    reg_nlp  = RegularizedNLPModel(qn_model, RootNormLhalf(λ))
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
    reg_nlp  = RegularizedNLPModel(qn_model, RootNormLhalf(λ))
    solver   = R2NSolver(reg_nlp)
    stats    = RegularizedExecutionStats(reg_nlp)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nlp, stats;
                                                x = x0, atol = atol, rtol = rtol, σk = σk,
                                                verbose = verbose, sub_kwargs = sub_kwargs, opnorm_maxiter = opnorm_maxiter)
    metrics = (
        name      = "R2N($(String(qn)))",
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
# ====== LANCEMENTS ======= #
#############################
results = NamedTuple[]

if :PANOC in CFG2.RUN_SOLVERS
    push!(results, run_panoc!(model, x0; λ = CFG2.LAMBDA_L0, maxit = CFG2.MAXIT_PANOC, tol = CFG2.TOL, verbose = CFG2.VERBOSE_PANOC))
end
if :TR in CFG2.RUN_SOLVERS
    push!(results, run_tr!(model, x0; λ = CFG2.LAMBDA_L0, qn = CFG2.QN_FOR_TR, atol = CFG2.TOL, rtol = CFG2.RTOL, verbose = CFG2.VERBOSE_RO, sub_kwargs = CFG2.SUB_KWARGS_R2N, opnorm_maxiter = CFG2.OPNORM_MAXITER))
end
if :R2N in CFG2.RUN_SOLVERS
    push!(results, run_r2n!(model, x0; λ = CFG2.LAMBDA_L0, qn = CFG2.QN_FOR_R2N, atol = CFG2.TOL, rtol = CFG2.RTOL,
                            verbose = CFG2.VERBOSE_RO, sub_kwargs = CFG2.SUB_KWARGS_R2N, σk = CFG2.SIGMAK_R2N, opnorm_maxiter = CFG2.OPNORM_MAXITER))
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
    println("accuracy (%)  = ", round(acc(residual(nls_train, m.solution)), digits=1))
end

println("\n")
print_config(CFG2)

if CFG2.PRINT_TABLE
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
    table_str = pretty_table(String, data;
           header = ["Method", "Status", "Time (s)", "#f", "#∇f", "#prox"],
           tf = tf_unicode,
           alignment = [:l, :c, :r, :r, :r, :r],
           crop = :none,
       )

    open("SVM-comparison-f.txt", "w") do io
        write(io, table_str)
    end

end