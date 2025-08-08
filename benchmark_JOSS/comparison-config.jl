module ComparisonConfig

Base.@kwdef mutable struct Config
    SEED::Int                 = 1234
    LAMBDA_L0::Float64        = 1.0
    TOL::Float64              = 1e-3
    RTOL::Float64             = 1e-3
    MAXIT_PANOC::Int          = 500
    VERBOSE_PANOC::Bool       = false
    VERBOSE_RO::Int           = 10
    RUN_SOLVERS::Vector{Symbol} = [:PANOC, :TR, :R2N]   # mutable
    QN_FOR_TR::Symbol         = :LSR1
    QN_FOR_R2N::Symbol        = :LBFGS
    SUB_KWARGS_R2N::NamedTuple = (; max_iter = 200)
    SIGMAK_R2N::Float64        = 1e5
    X0_SCALAR::Float64        = 0.1
    PRINT_TABLE::Bool         = true
end

# One global, constant *binding* to a mutable object = type stable & editable
const CFG = Config()


end # module