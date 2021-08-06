using Dates
using Logging

using PGFPlots
using TRNC

include("LMTR_TR-FitzHugh_datagen.jl")

# fancy logging
const fmt = Dates.@dateformat_str "yyyy-mm-dd HH:MM:SS:sss"

function my_metafmt(solver_name::AbstractString, level::Logging.LogLevel, _module, group, id, file, line)
  @nospecialize
  color = Logging.default_logcolor(level)
  prefix = Dates.format(Dates.now(), fmt) * ":" * solver_name
  suffix = ""
  Logging.Info <= level < Logging.Warn && return color, prefix, suffix
  _module !== nothing && (suffix *= "$(_module)")
  if file !== nothing
    suffix *= Base.contractuser(file)
    if line !== nothing
      suffix *= ":$(isa(line, UnitRange) ? "$(first(line))-$(last(line))" : line)"
    end
  end
  !isempty(suffix) && (suffix = "@ " * suffix)
  return color, prefix, suffix
end

function plot_results(xtr, Comp_pg, objdec, name = "tr-qr")
  F = simulate(xtr)
  a = Axis([Plots.Linear(1:size(F, 2), F[1, :], mark="none", legendentry="V"),
            Plots.Linear(1:size(F, 2), F[2, :], mark="none", legendentry="W"),
            Plots.Linear(1:size(F, 2), data[1, :], onlyMarks=true, markSize=1, mark="o", legendentry="V data"),
            Plots.Linear(1:size(F, 2), data[2, :], onlyMarks=true, markSize=1, mark="*", legendentry="W data"),
          ],
          xlabel="time",
          ylabel="voltage",
          legendStyle = "at={(1.0,1.0)}, anchor=north east, draw=none, font=\\scriptsize"
          )
  save("fh-$(name).pdf", a)

  b = Axis(Plots.Linear(1:length(Comp_pg), Comp_pg, mark="none"),
          xlabel="outer iterations",
          ylabel="inner iterations",
          ymode="log",
          )
  save("fh-inner-outer-$(name).pdf", b)

  c = Axis(Plots.Linear(1:length(objdec), objdec, mark = "none"),
          xlabel="\$ k^{th}\$  \$ \\nabla f \$ Call", 
          ylabel="Objective Value",
          ymode = "log"
        )
  save("fh-objdec-$(name).pdf",c)
end

outer_logger = ConsoleLogger(
                stderr,
                Logging.Info;
                meta_formatter = (args...) -> my_metafmt("TR", args...))

inner_logger = ConsoleLogger(
                stderr,
                Logging.Error;
                meta_formatter = (args...) -> my_metafmt("QR", args...))

data, simulate, resid, misfit = FH_smooth_term()
nls = ADNLSModel(resid, ones(5), 202)  # adbackend = ForwardDiff by default
nlp = LBFGSModel(ADNLPModel(misfit, ones(5)))

λ = 1.0
ϵ = 1.0e-6
h = NormL0(λ)
χ = NormLinf(1.0)
options = TRNCoptions(; maxIter = 1000, verbose = 10, ϵ = ϵ, β = 1e16, ν = 1.0e+2)

xtr, k, Fhist, Hhist, Comp_pg = with_logger(outer_logger) do
  TR(nlp, h, χ, options; subsolver_logger = inner_logger)
end

plot_results(xtr, Comp_pg[2,:], Fhist+Hhist, "tr-qr")

xtr, k, Fhist, Hhist, Comp_pg = with_logger(outer_logger) do
  LMTR(nls, h, χ, options; subsolver_logger = inner_logger)
end

plot_results(xtr, Comp_pg[2,:],Fhist+Hhist, "lmtr-qr")

reset!(nls)
xtr, k, Fhist, Hhist, Comp_pg = with_logger(outer_logger) do
  LM(nls, h, options; subsolver_logger = inner_logger)
end

plot_results(xtr, Comp_pg[2,:], Fhist+Hhist, "lm-qr")

