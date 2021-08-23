export AbstractExecutionStats,
  GenericExecutionStats, statsgetfield, statshead, statsline, getStatus, show_statuses

const STATUSES = Dict(
  :exception => "unhandled exception",
  :first_order => "first-order stationary",
  :acceptable => "solved to within acceptable tolerances",
  :max_eval => "maximum number of function evaluations",
  :max_iter => "maximum iteration",
  :max_time => "maximum elapsed time",
  :small_step => "step too small",
  :stalled => "stalled",
  :unbounded => "objective function may be unbounded from below",
  :unknown => "unknown",
  :user => "user-requested stop",
)

"""
    show_statuses()
Show the list of available statuses to use with `GenericExecutionStats`.
"""
function show_statuses()
  println("STATUSES:")
  for k in keys(STATUSES) |> collect |> sort
    v = STATUSES[k]
    @printf("  :%-14s => %s\n", k, v)
  end
end

abstract type AbstractExecutionStats end

"""
    GenericExecutionStats(status, F; ...)
A GenericExecutionStats is a struct for storing output information of solvers.
It contains the following fields:
- `status`: Indicates the output of the solver. Use `show_statuses()` for the full list;
- `solution`: The final approximation returned by the solver (default: `[]`);
- `objective`: The objective value at `solution` (default: `Inf`);
- `√ξ₁`: The stopping condtion at the `solution` (default: `Inf`);
- `Fhist`: Objective history of smooth function (default: `[]`);
- `Hhist`: Objective history of nonsmooth function (default: `[]`);
- `iter`: The number of iterations computed by the solver (default: `-1`);
- `elapsed_time`: The elapsed time computed by the solver (default: `Inf`);
- `counters::NLPModels.NLSCounters`: The Internal structure storing the number of functions evaluations;
- `solver_specific::Dict{Symbol,Any}`: A solver specific dictionary.
The `counters` variable is a copy of `F`'s counters, and `status` is mandatory on construction.
All other variables can be input as keyword arguments.
Notice that `GenericExecutionStats` does not compute anything, it simply stores.
"""
mutable struct GenericExecutionStats{T, V} <: AbstractExecutionStats
  status::Symbol
  solution::V # x
  objective::T # f(x) + h(x)
  ξ₁::Real # f(x) + h(x) - mₖ(s₁) = ξ₁
  Fhist::Vector{T} # objective history in f
  Hhist::Vector{T} # objective history in h
  SubsolverCounter::Vector{Int} # number of iterations the subsolver takes
  iter::Int
  counters::NLPModels.NLSCounters
  elapsed_time::Real
  solver_specific::Dict{Symbol, Any}
end

function GenericExecutionStats(
  status::Symbol,
  F::Union{Func, AbstractNLPModel{T, S}},
  H::ProximableFunction;
  solution::V = T[],
  kwargs...,
) where {S, T, V, Func <: Function}
  return GenericExecutionStats{Float64, V}(status, F, H; solution = solution, kwargs...)
end

function GenericExecutionStats{T, V}(
  status::Symbol,
  F::Union{Func, AbstractNLPModel},
  H::ProximableFunction;
  solution::V = T[],
  objective::T = T(Inf),
  ξ₁::T = T(Inf),
  Fhist::AbstractArray = V(undef, 0),
  Hhist::AbstractArray = V(undef, 0),
  SubsolverCounter::AbstractArray = V(undef, 0),
  iter::Int = -1,
  elapsed_time::Real = Inf,
  solver_specific::Dict{Symbol, Tsp} = Dict{Symbol, Any}(),
) where {T, V, Tsp, Func <: Function}
  if !(status in keys(STATUSES))
    @error "status $status is not a valid status. Use one of the following: " join(
      keys(STATUSES),
      ", ",
    )
    throw(KeyError(status))
  end
  c = NLSCounters()
  if F isa AbstractNLPModel
    for counter in fieldnames(Counters)
      setfield!(c.counters, counter, eval(Meta.parse("$counter"))(F))
    end
    if F isa AbstractNLSModel
      for counter in fieldnames(NLSCounters)
        counter == :counters && continue
        setfield!(c, counter, eval(Meta.parse("$counter"))(F))
      end
    end
  end
  return GenericExecutionStats{T, V}(
    status,
    solution,
    objective,
    ξ₁,
    Fhist,
    Hhist,
    SubsolverCounter,
    iter,
    c,
    elapsed_time,
    solver_specific,
  )
end

import Base.show, Base.print, Base.println

function show(io::IO, stats::AbstractExecutionStats)
  show(io, "Execution stats: $(getStatus(stats))")
end

function disp_vector(io::IO, x::AbstractArray)
  if length(x) == 0
    print(io, "∅")
  elseif length(x) <= 5
    Base.show_delim_array(io, x, "[", " ", "]", false)
  else
    Base.show_delim_array(io, x[1:4], "[", " ", "", false)
    print(io, " ⋯ $(x[end])]")
  end
end

function print(io::IO, stats::GenericExecutionStats; showvec::Function = disp_vector)
  # TODO: Show evaluations
  println(io, "Generic Execution stats")
  println(io, "  status: " * getStatus(stats))
  println(io, "  objective value: ", stats.objective)
  println(io, "  √ξ₁: ", stats.ξ₁)
  print(io, "  solution: ")
  showvec(io, stats.solution)
  println(io, "")
  length(stats.Fhist) > 0 &&
    (print(io, "  Fhist: "); showvec(io, stats.Fhist); println(io, ""))
  length(stats.Hhist) > 0 &&
    (print(io, "  Hhist: "); showvec(io, stats.Hhist); println(io, ""))
  println(io, "  iterations: ", stats.iter)
  println(io, "  elapsed time: ", stats.elapsed_time)
  if length(stats.solver_specific) > 0
    println(io, "  solver specific:")
    for (k, v) in stats.solver_specific
      @printf(io, "    %s: ", k)
      if v isa Vector
        showvec(io, v)
      else
        show(io, v)
      end
      println(io, "")
    end
  end
end
print(stats::GenericExecutionStats; showvec::Function = disp_vector) =
  print(Base.stdout, stats, showvec = showvec)
println(io::IO, stats::GenericExecutionStats; showvec::Function = disp_vector) =
  print(io, stats, showvec = showvec)
println(stats::GenericExecutionStats; showvec::Function = disp_vector) =
  print(Base.stdout, stats, showvec = showvec)

const headsym = Dict(
  :status => "  Status",
  :iter => "   Iter",
  :neval_obj => "   #obj",
  :neval_grad => "  #grad",
  :neval_cons => "  #cons",
  :neval_jcon => "  #jcon",
  :neval_jgrad => " #jgrad",
  :neval_jac => "   #jac",
  :neval_jprod => " #jprod",
  :neval_jtprod => "#jtprod",
  :neval_hess => "  #hess",
  :neval_hprod => " #hprod",
  :neval_jhprod => "#jhprod",
  :objective => "              f+h",
  :ξ₁ => "           f+h-mₖ",
  :elapsed_time => "  Elaspsed time",
)

function statsgetfield(stats::AbstractExecutionStats, name::Symbol)
  t = Int
  if name == :status
    v = getStatus(stats)
    t = String
  elseif name in fieldnames(NLPModels.NLSCounters)
    v = getfield(stats.counters, name)
  elseif name in fieldnames(NLPModels.Counters)
    if stats.counters isa NLPModels.Counters
      v = getfield(stats.counters, name)
    else
      v = getfield(stats.counters.counters, name)
    end
  elseif name in fieldnames(typeof(stats))
    v = getfield(stats, name)
    t = fieldtype(typeof(stats), name)
  else
    error("Unknown field $name")
  end
  if t <: Int
    @sprintf("%7d", v)
  elseif t <: Real
    @sprintf("%15.8e", v)
  else
    @sprintf("%8s", v)
  end
end

function statshead(line::Array{Symbol})
  return join([headsym[x] for x in line], "  ")
end

function statsline(stats::AbstractExecutionStats, line::Array{Symbol})
  return join([statsgetfield(stats, x) for x in line], "  ")
end

function getStatus(stats::AbstractExecutionStats)
  return STATUSES[stats.status]
end