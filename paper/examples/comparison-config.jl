module ComparisonConfig

Base.@kwdef mutable struct Config
    SEED::Int                 = 1234
    LAMBDA_L0::Float64        = 1.0
    TOL::Float64              = 1e-4
    RTOL::Float64             = 1e-4
    MAXIT_PANOC::Int          = 10000
    VERBOSE_PANOC::Bool       = false
    VERBOSE_RO::Int           = 0
    RUN_SOLVERS::Vector{Symbol} = [:PANOC, :TR, :R2N]   # mutable
    QN_FOR_TR::Symbol         = :LSR1
    QN_FOR_R2N::Symbol        = :LBFGS
    SUB_KWARGS_R2N::NamedTuple = (; max_iter = 200)
    PRINT_TABLE::Bool         = true
end

# One global, constant *binding* to a mutable object = type stable & editable
const CFG = Config(QN_FOR_R2N=:LSR1)
const CFG2 = Config(RUN_SOLVERS = [:LM, :TR, :R2N], QN_FOR_TR = :LBFGS)

end # module