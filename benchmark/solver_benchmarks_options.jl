function solver_benchmark_profile_values()
  [(:elapsed_time, "CPU Time"), (:neval_obj, "# Objective Evals"), (:neval_grad, "# Gradient Evals"), (:iter, "# Iterations")]
end

function solver_benchmark_table_values()
  return [(:name, "Name"), (:objective, "f(x)"), (:elapsed_time, "Time"), (:neval_obj, "Obj Evals"), (:neval_grad, "Grad Evals"), (:iter, "Iterations")]
end
