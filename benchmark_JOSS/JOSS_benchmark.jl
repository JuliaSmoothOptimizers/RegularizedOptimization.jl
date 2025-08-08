
#############################
# ======== IMPORTS ======== #
#############################
using Random, LinearAlgebra
using ProximalOperators, ProximalCore, ProximalAlgorithms
using ADNLPModels, NLPModels, NLPModelsModifiers
using RegularizedOptimization, RegularizedProblems
using DifferentialEquations, SciMLSensitivity

include("comparison-config.jl")
using .ComparisonConfig: CFG

include("Bench-utils.jl")
using .BenchUtils

function print_config(cfg)
    println("Configuration:")
    println("  SEED            = $(cfg.SEED)")
    println("  LAMBDA_L0       = $(cfg.LAMBDA_L0)")
    println("  TOL             = $(cfg.TOL)")
    println("  RTOL            = $(cfg.RTOL)")
    println("  MAXIT_PANOC     = $(cfg.MAXIT_PANOC)")
    println("  VERBOSE_PANOC   = $(cfg.VERBOSE_PANOC)")
    println("  VERBOSE_RO      = $(cfg.VERBOSE_RO)")
    println("  RUN_SOLVERS     = $(cfg.RUN_SOLVERS)")
    println("  QN_FOR_TR       = $(cfg.QN_FOR_TR)")
    println("  QN_FOR_R2N      = $(cfg.QN_FOR_R2N)")
    println("  SUB_KWARGS_R2N  = $(cfg.SUB_KWARGS_R2N)")
    println("  SIGMAK_R2N      = $(cfg.SIGMAK_R2N)")
    println("  X0_SCALAR       = $(cfg.X0_SCALAR)")
    println("  PRINT_TABLE     = $(cfg.PRINT_TABLE)")   
end

#############################
# ===== PROBLÈME (FH) ===== #
#############################
Random.seed!(CFG.SEED)

# Si tu as fh_model() (wrapper perso) qui renvoie (model, misfit?, x*)
if @isdefined fh_model
    model, _, x_true = fh_model()
else
    # Fallback: construit le modèle depuis RegularizedProblems
    _, _, _, misfit, _ = RegularizedProblems.FH_smooth_term()
    model = ADNLPModel(misfit, x0; matrix_free = true)
    x_true = nothing
end

x0 = CFG.X0_SCALAR .* ones(length(model.meta.x0))
obj(model, x0)

#############################
# ======= PANOC run ======= #
#############################
function run_panoc!(model, x0; λ = 1.0, maxit = 500, tol = 1e-3, verbose = false)
   # BenchUtils.make_adnlp_compatible!()
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

function run_tr!(model, x0; λ = 1.0, qn = :LSR1, atol = 1e-3, rtol = 1e-3, verbose = 0, sub_kwargs = (;))
    qn_model = ensure_qn(model, qn)
    reset!(qn_model)  # reset des compteurs
    reg_nlp  = RegularizedNLPModel(qn_model, NormL0(λ))
    solver   = TRSolver(reg_nlp)
    stats    = RegularizedExecutionStats(reg_nlp)
    obj(qn_model, x0)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nlp, stats;
                                                x = x0, atol = atol, rtol = rtol, verbose = verbose, opnorm_maxiter = 1, sub_kwargs = sub_kwargs)
    metrics = (
        name      = "TR($(String(qn)))",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_obj(qn_model),
        gevals    = neval_grad(qn_model),
        proxcalls = missing, #stats.solver_specific[:prox_evals],
        solution  = stats.solution,
    )
    return metrics
end

#############################
# ======== R2N run ======== #
#############################
function run_r2n!(model, x0; λ = 1.0, qn = :LBFGS, atol = 1e-3, rtol = 1e-3, verbose = 0, sub_kwargs = (;), σk = 1e5)
    qn_model = ensure_qn(model, qn)
    reset!(qn_model)
    reg_nlp  = RegularizedNLPModel(qn_model, NormL0(λ))
    solver   = R2NSolver(reg_nlp)
    stats    = RegularizedExecutionStats(reg_nlp)
    t = @elapsed RegularizedOptimization.solve!(solver, reg_nlp, stats;
                                                x = x0, atol = atol, rtol = rtol, σk = σk,
                                                verbose = verbose, sub_kwargs = sub_kwargs, opnorm_maxiter = 1)
    metrics = (
        name      = "R2N($(String(qn)))",
        status    = string(stats.status),
        time      = t,
        iters     = get(stats.solver_specific, :outer_iter, missing),
        fevals    = neval_obj(qn_model),
        gevals    = neval_grad(qn_model),
        proxcalls = missing, #stats.solver_specific[:prox_evals],
        solution  = stats.solution,
    )
    return metrics
end

#############################
# ====== LANCEMENTS ======= #
#############################
results = NamedTuple[]

if :PANOC in CFG.RUN_SOLVERS
    push!(results, run_panoc!(model, x0; λ = CFG.LAMBDA_L0, maxit = CFG.MAXIT_PANOC, tol = CFG.TOL, verbose = CFG.VERBOSE_PANOC))
end
if :TR in CFG.RUN_SOLVERS
    push!(results, run_tr!(model, x0; λ = CFG.LAMBDA_L0, qn = CFG.QN_FOR_TR, atol = CFG.TOL, rtol = CFG.RTOL, verbose = CFG.VERBOSE_RO, sub_kwargs = CFG.SUB_KWARGS_R2N,))
end
if :R2N in CFG.RUN_SOLVERS
    push!(results, run_r2n!(model, x0; λ = CFG.LAMBDA_L0, qn = CFG.QN_FOR_R2N, atol = CFG.TOL, rtol = CFG.RTOL,
                            verbose = CFG.VERBOSE_RO, sub_kwargs = CFG.SUB_KWARGS_R2N, σk = CFG.SIGMAK_R2N))
end


#############################
# ===== AFFICHAGE I/O ===== #
#############################
if x_true !== nothing
    println("=== True solution (≈) ===")
    println(x_true)
end

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
        println("  # prox appels = ", m.proxcalls)
    end
    println("  solution (≈) = ", m.solution)
end

if CFG.PRINT_TABLE
    println("\nRésumé :")
    header = (
        rpad("Méthode", 14) *
        rpad("statut", 12) *
        rpad("temps(s)", 10) *
        rpad("#f", 8) *
        rpad("#∇f", 8) *
        rpad("#prox", 8)
    )
    println(header)
    for m in results
        prox = m.proxcalls === missing ? "-" : string(m.proxcalls)
        line = (
            rpad(m.name, 14) *
            rpad(string(m.status), 12) *
            rpad(string(round(m.time, digits=4)), 10) *
            rpad(string(m.fevals), 8) *
            rpad(string(m.gevals), 8) *
            rpad(prox, 8)
        )
        println(line)
    end
end
