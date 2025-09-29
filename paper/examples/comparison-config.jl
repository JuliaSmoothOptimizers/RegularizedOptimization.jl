module ComparisonConfig

Base.@kwdef mutable struct Config
    SEED::Int                 = 1234
    LAMBDA_L0::Float64        = 1.0
    TOL::Float64              = 1e-3
    RTOL::Float64             = 1e-3
    MAXIT_PANOC::Int          = 10000
    VERBOSE_PANOC::Bool       = false
    VERBOSE_RO::Int           = 0
    RUN_SOLVERS::Vector{Symbol} = [:PANOC, :TR, :R2N]   # mutable
    QN_FOR_TR::Symbol         = :LSR1
    QN_FOR_R2N::Symbol        = :LBFGS
    SUB_KWARGS_R2N::NamedTuple = (; max_iter = 200)
    SIGMAK_R2N::Float64        = 1e5
    X0_SCALAR::Float64        = 0.1
    PRINT_TABLE::Bool         = true
    OPNORM_MAXITER::Int       = 20
    HESSIAN_SCALE::Float64    = 1e-4
    M_MONOTONE::Int           = 10  # for nonmonotone R2N
end

# One global, constant *binding* to a mutable object = type stable & editable
const CFG = Config(QN_FOR_TR = :LBFGS)
const CFG2 = Config(SIGMAK_R2N=eps()^(1/3), TOL = 1e-4, RTOL = 1e-4, QN_FOR_R2N=:LSR1, M_MONOTONE=1)
const CFG3 = Config(SIGMAK_R2N=1e3, TOL = 1e-4, RTOL = 1e-4,  RUN_SOLVERS = [:LM, :TR, :R2N], QN_FOR_TR = :LBFGS)

end # module