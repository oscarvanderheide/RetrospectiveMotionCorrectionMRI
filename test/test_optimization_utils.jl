using MotionCorrectedMRI, LinearAlgebra, Flux, Test
import Flux.Optimise: Optimiser, update!

# Setting linear system
T = Float32
Q = qr(randn(T, 100, 100)).Q
A = Q*diagm(T(1).+T(0.1)*randn(T,100))*Q'
b = randn(T, 100)
xtrue = A\b

# FISTA
x0 = randn(T, 100)
L = spectral_radius(A'*A, randn(T,100); niter=1000)
prox(x, λ) = x
Nesterov = true
# Nesterov = false
opt_fista = OptimiserFISTA(L, prox; Nesterov=Nesterov, reset_counter=20)
niter = 100
fval_fista = Array{T,1}(undef, niter)
x = deepcopy(x0)
for i = 1:niter
    r = A*x-b
    fval_fista[i] = 0.5*norm(r)^2
    local g = A'*r
    update!(opt_fista, x, g)
end
@test x ≈ xtrue rtol=1e-5