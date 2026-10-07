using LinearAlgebra, AbstractProximableFunctions, AbstractLinearOperators, Test, Random
Random.seed!(123)

# Random input
T = Float64
x = randn(T, 2, 3, 4)

# Zero set
C = zero_set(T, x .> 0)

# Projection test
y = proj(x, C)
@test norm(y[x .> 0]) ≈ 0 rtol=T(1e-6)
@test proj(y, C) ≈ y rtol=T(1e-6)

# Gradient test (indicator)
g = indicator(C)
fun = prox_objfun(g, T(0.1))
y = randn(T, 2, 3, 4)
@test test_grad(fun, y; step=1e-4, rtol=1e-3)

fun = proj_objfun(g, T(0.1))
y = randn(T, 2, 3, 4)
@test test_grad(fun, y; step=1e-4, rtol=1e-3)