module ComparisonConfig

Base.@kwdef mutable struct Config
    SEED::Int                 = 1234
    LAMBDA_L0::Float64        = 1.0
    TOL::Float64              = 1e-3
    RTOL::Float64             = 1e-3
    MAXIT_PANOC::Int          = 500
    VERBOSE_PANOC::Bool       = false
    VERBOSE_RO::Int           = 0
    RUN_SOLVERS::Vector{Symbol} = [:PANOC, :TR, :R2N]   # mutable
    QN_FOR_TR::Symbol         = :LSR1
    QN_FOR_R2N::Symbol        = :LBFGS
    SUB_KWARGS_R2N::NamedTuple = (; max_iter = 200)
    SIGMAK_R2N::Float64        = 1e5
    X0_SCALAR::Float64        = 0.1
    PRINT_TABLE::Bool         = true
    OPNORM_MAXITER::Int       = 4
    HESSIAN_SCALE::Float64   = 1e-4
end

# One global, constant *binding* to a mutable object = type stable & editable
const CFG = Config()
const CFG2 = Config(SIGMAK_R2N=eps()^(1/3), TOL = 1e-4, RTOL = 1e-4)
const CFG3 = Config(SIGMAK_R2N=1e3, TOL = 1e-4, RTOL = 1e-4,  RUN_SOLVERS = [:LM, :TR, :R2N])

end # module