using BenchmarkTools
using RegularizedProblems, RegularizedOptimization

const SUITE = BenchmarkGroup()

bpdn_l0, _ = setup_bpdn_l0()
bpdn_l1, _ = setup_bpdn_l1()
bpdn_B0, _ = setup_bpdn_B0()

for solver ∈ (R2,)
  solver_name = string(solver)
  SUITE[solver_name] = BenchmarkGroup()
  SUITE[solver_name]["bpdn_l0"] = @benchmarkable $solver($bpdn_l0)
  SUITE[solver_name]["bpdn_l1"] = @benchmarkable $solver($bpdn_l1)
  SUITE[solver_name]["bpdn_B0"] = @benchmarkable $solver($bpdn_B0)
end