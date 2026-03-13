using Random
using RegularizedProblems, RegularizedOptimization, SolverBenchmark

# Options for the run_solver_benchmarks function.
# These functions specify the values to be used in the performance profiles and tables.
function Main.solver_benchmark_profile_values()
  [(:elapsed_time, "CPU Time"), (:neval_obj, "# Objective Evals"), (:neval_grad, "# Gradient Evals"), (:iter, "# Iterations")]
end

function Main.solver_benchmark_table_values()
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
    )
)

stats = bmark_solvers(solvers, problem_list)

