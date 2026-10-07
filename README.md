# AbstractProximableFunctions

Set of abstractions and utilities for computing proximal and projection operators of convex functionals

Example: mix and match convex differentiable objectives, convex non-differentiable terms, and hard constraints!
```
using LinearAlgebra, AbstractLinearOperators, AbstractProximableFunctions

# Setup differentiable objective
A = linear_operator(Float32, (64, 64), (64, 64), x->2*x, y->2*y)
y = randn(Float32, 64, 64)
obj = leastsquares_misfit(A, y)

# Proximal function
g = norm(Float32, 2, 1) # l1-norm for 2-D images

# Hard constraints
C = zero_set(Float32, randn(Float32, 64, 64) .> 0f0)

# Setup FISTA solver
ρ = 1.01f0*spectral_radius(A*A'; niter=10)
opt_inner = FISTA_options(1f0; Nesterov=true, niter=10, reset_counter=20, verbose=false, fun_history=false)
opt_outer = FISTA_options(ρ; Nesterov=true, niter=20, reset_counter=20, verbose=true, fun_history=false)

# Iterative solution
x0 = zeros(Float32, size(y))
x = argmin(obj+set_options(g+indicator(C), opt_inner), x0, opt_outer)

# Checking solution
x ∈ C
```

This package is highly indebted to [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) and related projects.