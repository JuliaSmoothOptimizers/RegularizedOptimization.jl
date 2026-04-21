"""
    @wrappedallocs(expr)

Given an expression, this macro wraps that expression inside a new function
which will evaluate that expression and measure the amount of memory allocated
by the expression. Wrapping the expression in a new function allows for more
accurate memory allocation detection when using global variables (e.g. when
at the REPL).

This code is based on that of https://github.com/JuliaAlgebra/TypedPolynomials.jl/blob/master/test/runtests.jl

You can use this macro in a unit test to verify that a function does not
allocate:

```
@test @wrappedallocs(x + y) == 0
```
"""
macro wrappedallocs(expr)
  kwargs = [a for a in expr.args if isa(a, Expr)]
  args = [a for a in expr.args if isa(a, Symbol)]

  argnames = [gensym() for a in args]
  kwargs_dict = Dict{Symbol, Any}(a.args[1] => a.args[2] for a in kwargs if a.head == :kw)
  quote
    function g($(argnames...); kwargs_dict...)
      $(Expr(expr.head, argnames..., kwargs...)) # Call the function twice to make the allocated macro more stable
      @allocated $(Expr(expr.head, argnames..., kwargs...))
    end
    $(Expr(:call, :g, [esc(a) for a in args]...))
  end
end

# Construct the rosenbrock problem.

function rosenbrock_f(x::Vector{T}) where {T <: Real}
  100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
end

function rosenbrock_grad!(gx::Vector{T}, x::Vector{T}) where {T <: Real}
  gx[1] = -400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1])
  gx[2] = 200 * (x[2] - x[1]^2)
end

function rosenbrock_hv!(hv::Vector{T}, x::Vector{T}, v::Vector{T}; obj_weight = 1.0) where {T}
  hv[1] = (1200 * x[1]^2 - 400 * x[2] + 2) * v[1] - 400 * x[1] * v[2]
  hv[2] = -400 * x[1] * v[1] + 200 * v[2]
end

function construct_rosenbrock_nlp()
  return NLPModel(zeros(2), rosenbrock_f, grad = rosenbrock_grad!, hprod = rosenbrock_hv!)
end
