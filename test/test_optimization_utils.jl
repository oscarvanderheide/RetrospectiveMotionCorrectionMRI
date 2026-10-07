using AbstractProximableFunctions, LinearAlgebra, Test, AbstractLinearOperators

rtol = 1e-5

# Setting linear system
T = Float64
Q = qr(randn(T, 100, 100)).Q
A = Q*diagm(T(1).+T(0.1)*randn(T,100))*Q'
b = randn(T, 100)
xtrue = A\b

# Via minimize routine
g = zero_prox(T, 1)
x0 = randn(T, 100)
L = T(1.1)*spectral_radius(A'*A; niter=100)
niter = 100
Nesterov = true
# Nesterov = false
opt_fista = FISTA_options(L; Nesterov=Nesterov, niter=niter, reset_counter=10, verbose=false, fun_history=true)
Aop = linear_operator(T, size(xtrue), size(xtrue), x->A*x, y->A'*y)
f = leastsquares_misfit(Aop, b)
x = argmin(f+g, x0, opt_fista)
fval_fista = fun_history(opt_fista)
@test x ≈ xtrue rtol=rtol

# Via least-squares routines
x = leastsquares_solve(Aop, b, g, x0, opt_fista)
@test x ≈ xtrue rtol=rtol