export GD_problem, GD_setting, GD_solver

struct GD_problem{F <: Function, prox<: Function, A <: AbstractVecOrMat{<:Real}, R <: Real, B<: Real}
    
    # problem structure, contains information regarding the problem
    
    f::F # the objective function
    prox:: prox #the prox function 
    x0::A # the intial condition
    ν::R # the stepsize
    λ::B
    
end

struct GD_setting
    
    # user settings to solve the problem using Gradient Descent
    
    ν::Float64 # the step size
    maxit::Int64 # maximum number of iteration
    tol::Float64 # tolerance, i.e., if ||∇f(x)|| ≤ tol, we take x to be an optimal solution
    verbose::Bool # whether to print information about the iterates
    freq::Int64 # how often print information about the iterates

    # constructor for the structure, so if user does not specify any particular values, 
    # then we create a GD_setting object with default values
    function GD_setting(; ν = 1, maxit = 1000, tol = 1e-8, verbose = false, freq = 10)
        new(ν, maxit, tol, verbose, freq)
    end
    
end

mutable struct GD_state#{T <: AbstractVecOrMat{<: Real}, I <: Integer, R <: Real} # contains information regarding one iterattion sequence
    
    x#::T # iterate x_n
    x⁻#::T #previous iterate 
    f_x#::T #function value 
    ∇f_x#::T # one gradient ∇f(x_n)
    ν#::T # stepsize
    λ#::T #h(x) regularizer  
    n#::I # iteration counter
    
end

function GD_state(problem::GD_problem)
    
    # a constructor for the struct GD_state, it will take the problem data and create one state containing all 
    # the iterate information, current state of the gradient etc so that we can start our gradient descent scheme
    
    # unpack information from iter which is GD_iterable type
    x0 = copy(problem.x0) # to be safe
    f = problem.f
    ν = problem.ν
    λ = problem.λ
    f_x, ∇f_x = f(x0)
    n = 1
    
    return GD_state(x0, x0, f_x, ∇f_x, ν, λ, n)
    
end


function GD_iteration!(problem::GD_problem, state::GD_state)
    
    # this is the main iteration function, that takes the problem information, and the previous state, 
    # and create the new state using Gradient Descent algorithm
    
    # unpack the current state information
    x_n = state.x
    ∇f_x_n = state.∇f_x
    ν_n = state.ν
    λ = state.λ
    n = state.n
    
    # compute the next state
    x_n_plus_1 = x_n - ν_n*∇f_x_n
    # prox projection
    x_n_plus_1 = problem.prox(x_n_plus_1, λ*ν_n)
    
    # now load the computed values in the state
    state.x⁻ = state.x
    state.x = x_n_plus_1
    f_x, state.∇f_x = problem.f(x_n_plus_1)
    # state.ν = 1/(n+1)
    state.n = n+1
    
    # done computing return the new state
    return state
    
end


## The solver function

function GD_solver(problem::GD_problem, setting::GD_setting)
    
    # this is the function that the end user will use to solve a particular problem, internally it is using the previously defined types and functions to run Gradient Descent Scheme
    # create the intial state
    state = GD_state(problem::GD_problem)
    ## time to run the loop
    while  (state.n < setting.maxit) & (norm(state.∇f_x, Inf) > setting.tol)
        # compute a new state
        state =  GD_iteration!(problem, state)
        # print information if verbose = true
        if setting.verbose == true
            if mod(state.n, setting.freq) == 0
                @info "iteration = $(state.n) | obj val = $(problem.f(state.x)[1]) | gradient norm = $(norm(state.∇f_x, Inf))"
            end
        end
    end
    
    # print information regarding the final state
    
    # @info "final iteration = $(state.n) | final obj val = $(problem.f(state.x)[1]) | final gradient norm = $(norm(state.∇f_x, Inf))"
    return state
    
end