using NLPModels,
  NLPModelsModifiers,
  RegularizedProblems,
  RegularizedOptimization,
  SolverCore, CUTEst, SparseMatricesCOO

nlp = LBFGSModel(CUTEstModel("MSS1"), damped = true, scaling = false, σ₂ = 0.1)
tol = 1e-3
max_time = 100.0

stats = L2Penalty(
      nlp,
      max_time=max_time,
      max_iter=typemax(Int64),
      max_eval=typemax(Int64),
      subsolver = R2Solver,
      atol=0.0,
      rtol=0.0,
      neg_tol=sqrt(tol),
      τ=500.0,
      callback=(nlp, solver, stats) -> begin
        if stats.primal_feas/stats.solver_specific[:theta] > typeof(stats.primal_feas)(100) && stats.solver_specific[:theta] < tol
          stats.status = :infeasible
        end
        if stats.primal_feas < tol && stats.dual_feas < tol
          stats.status = :user          
        end
      end,
      sub_callback=(nlp, solver, stats) -> begin
        isa(solver, R2NSolver) && (stats.dual_feas = norm(solver.s1)*stats.solver_specific[:sigma_cauchy])
        isa(solver, R2Solver) && (stats.dual_feas = norm(solver.s)*stats.solver_specific[:sigma])
        if stats.dual_feas < tol
          stats.status = :user 
        end
      end,
      verbose = 1
  )

println(stats)

finalize(nlp)
finalize(nlp.model)
