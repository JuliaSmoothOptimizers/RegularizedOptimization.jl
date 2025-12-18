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

# Construct the brock-rosenberg problem.
