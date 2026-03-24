using Random
using NLPModelsModifiers, RegularizedProblems, RegularizedOptimization, SolverBenchmark

# Options for the run_solver_benchmarks function.
# These functions specify the values to be used in the performance profiles and tables.
function JSOBenchmarks.solver_benchmark_profile_values()
  [(:elapsed_time, "CPU Time"), (:neval_obj, "# Objective Evals"), (:neval_grad, "# Gradient Evals"), (:iter, "# Iterations")]
end

function JSOBenchmarks.solver_benchmark_table_values()
  return [(:name, "Name"), (:objective, "f(x)"), (:elapsed_time, "Time"), (:neval_obj, "Obj Evals"), (:neval_grad, "Grad Evals"), (:iter, "Iterations")]
end

Random.seed!(0)

problem_list = []

n_bpdn = 10
for _ in 1:n_bpdn
  bpdn_l0, _ = setup_bpdn_l0()
  bpdn_l1, _ = setup_bpdn_l1()
  bpdn_B0, _ = setup_bpdn_B0()
  push!(problem_list, bpdn_l0, bpdn_l1, bpdn_B0)
end

n_lasso = 10
for _ in 1:n_lasso
  lasso_l12, _ = setup_group_lasso_l12()
  push!(problem_list, lasso_l12)
end

n_nnmf = 5
for _ in 1:n_nnmf
  nnmf_l0, _ = setup_nnmf_l0()
  nnmf_l1, _ = setup_nnmf_l1()
  push!(problem_list, nnmf_l0, nnmf_l1)
end

n_qp = 10
for _ in 1:n_qp
  qp_l1, _ = setup_qp_rand_l1()
  push!(problem_list, qp_l1)
end

solvers = Dict(
  # R2
  :R2_imprecise => 
    reg_nlp -> R2(
      reg_nlp,
      verbose = 1,
      atol = 1e-3,
      rtol = 1e-3,
    ),
  :R2_default => 
    reg_nlp -> R2(
      reg_nlp,
      verbose = 1,
    ),
  
  # R2N with BFGS
  :R2N_bfgs_imprecise =>
    reg_nlp -> R2N(
      RegularizedNLPModel(LBFGSModel(reg_nlp.model), reg_nlp.h, reg_nlp.selected),
      verbose = 1,
      atol = 1e-3,
      rtol = 1e-3,
    ),
  :R2N_bfgs_default =>
    reg_nlp -> R2N(
      RegularizedNLPModel(LBFGSModel(reg_nlp.model), reg_nlp.h, reg_nlp.selected),
      verbose = 1,
    ),
  
  # R2N with SR1
  :R2N_sr1_imprecise =>
    reg_nlp -> R2N(
      RegularizedNLPModel(LSR1Model(reg_nlp.model), reg_nlp.h, reg_nlp.selected),
      verbose = 1,
      atol = 1e-3,
      rtol = 1e-3,
    ),
  :R2N_sr1_default =>
    reg_nlp -> R2N(
      RegularizedNLPModel(LSR1Model(reg_nlp.model), reg_nlp.h, reg_nlp.selected),
      verbose = 1,
    ),

  # TR with BFGS
  :TR_bfgs_imprecise =>
    reg_nlp -> TR(
      RegularizedNLPModel(LBFGSModel(reg_nlp.model), reg_nlp.h, reg_nlp.selected),
      verbose = 1,
      atol = 1e-3,
      rtol = 1e-3,
    ),
  :TR_bfgs_default =>
    reg_nlp -> TR(
      RegularizedNLPModel(LBFGSModel(reg_nlp.model), reg_nlp.h, reg_nlp.selected),
      verbose = 1,
    ),
  
  # TR with SR1
  :TR_sr1_imprecise =>
    reg_nlp -> TR(
      RegularizedNLPModel(LSR1Model(reg_nlp.model), reg_nlp.h, reg_nlp.selected),
      verbose = 1,
      atol = 1e-3,
      rtol = 1e-3,
    ),
  :TR_sr1_default =>
    reg_nlp -> TR(
      RegularizedNLPModel(LSR1Model(reg_nlp.model), reg_nlp.h, reg_nlp.selected),
      verbose = 1,
    ),

  # R2DH
  :R2DH_imprecise =>
    reg_nlp -> R2DH(
      reg_nlp,
      verbose = 1,
      atol = 1e-3,
      rtol = 1e-3,
    ),
  :R2DH_default =>
    reg_nlp -> R2DH(
      reg_nlp,
      verbose = 1,
    ),
  
  # TRDH
  :TRDH_imprecise =>
    reg_nlp -> TRDH(
      reg_nlp,
      verbose = 1,
      atol = 1e-3,
      rtol = 1e-3,
    ),
  :TRDH_default =>
    reg_nlp -> TRDH(
      reg_nlp,
      verbose = 1,
    ),
)

stats = bmark_solvers(solvers, problem_list)

