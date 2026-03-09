using RegularizedProblems, RegularizedOptimization, SolverBenchmark

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
      verbose = 0,
      atol = 1e-6,
      rtol = 1e-6,
    ),
  :R2_imprecise => 
    reg_nlp -> R2(
      reg_nlp,
      verbose = 0,
      atol = 1e-3,
      rtol = 1e-3,
    )
)

stats = bmark_solvers(solvers, problem_list)

