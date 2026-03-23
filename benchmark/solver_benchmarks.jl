using Random
using RegularizedProblems, RegularizedOptimization, SolverBenchmark

# Options for the run_solver_benchmarks function.
# These functions specify the values to be used in the performance profiles and tables.
function JSOBenchmarks.solver_benchmark_profile_values()
  [(:elapsed_time, "CPU Time"), (:neval_obj, "# Objective Evals"), (:neval_grad, "# Gradient Evals"), (:iter, "# Iterations")]
end

function JSOBenchmarks.solver_benchmark_table_values()
  return [(:name, "Name"), (:objective, "f(x)"), (:elapsed_time, "Time"), (:neval_obj, "Obj Evals"), (:neval_grad, "Grad Evals"), (:iter, "Iterations")]
end

Random.seed!(0)

bpdn_l0, _ = setup_bpdn_l0()
bpdn_l1, _ = setup_bpdn_l1()
bpdn_B0, _ = setup_bpdn_B0()

problem_list = [
  bpdn_l0,
  bpdn_l1,
  bpdn_B0,
]

solvers = Dict(
  # R2
  :R2_precise => 
    reg_nlp -> R2(
      reg_nlp,
      verbose = 1,
      atol = 1e-6,
      rtol = 1e-6,
    ),
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
  :R2N_bfgs_precise => 
    reg_nlp -> R2N(
      RegularizedNLPModel(LBFGSModel(reg_nlp.model), reg_nlp.h, reg_nlp.selected),
      verbose = 1,
      atol = 1e-6,
      rtol = 1e-6,
    ),
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
  :R2N_sr1_precise => 
    reg_nlp -> R2N(
      RegularizedNLPModel(LSR1Model(reg_nlp.model), reg_nlp.h, reg_nlp.selected),
      verbose = 1,
      atol = 1e-6,
      rtol = 1e-6,
    ),
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
  :TR_bfgs_precise => 
    reg_nlp -> TR(
      RegularizedNLPModel(LBFGSModel(reg_nlp.model), reg_nlp.h, reg_nlp.selected),
      verbose = 1,
      atol = 1e-6,
      rtol = 1e-6,
    ),
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
  :TR_sr1_precise => 
    reg_nlp -> TR(
      RegularizedNLPModel(LSR1Model(reg_nlp.model), reg_nlp.h, reg_nlp.selected),
      verbose = 1,
      atol = 1e-6,
      rtol = 1e-6,
    ),
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
  :R2DH_precise => 
    reg_nlp -> R2DH(
      reg_nlp,
      verbose = 1,
      atol = 1e-6,
      rtol = 1e-6,
    ),
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
  :TRDH_precise => 
    reg_nlp -> TRDH(
      reg_nlp,
      verbose = 1,
      atol = 1e-6,
      rtol = 1e-6,
    ),
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

