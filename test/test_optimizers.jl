using UtilitiesForMRI, LinearAlgebra, Flux
import Flux.Optimise: Optimiser, update!

# Setting linear system
T = Float64
A = Matrix{T}(I, 100, 100)+T(0.01)*randn(T, 100, 100)
b = randn(T, 100)
xtrue = A\b

# Anderson acceleration
hist_size = 20
β = T(0.9)
lr = T(1e-1)
opt_and = Anderson(; lr=lr, hist_size=hist_size, β=β)

# Minimization
niter = 100
x = randn(T, 100)
g = similar(x)
fval_and = Array{T,1}(undef, niter)
for i = 1:niter
    r = A*x-b
    fval_and[i] = norm(r)^2/2
    g .= A'*r
    update!(opt_and, x, g)
end
err = norm(x-xtrue)/norm(xtrue)

# Setting linear system
T = Float32
Q = qr(randn(T, 100, 100)).Q
A = Q*diagm(T(1).+T(0.1)*randn(T,100))*Q'
b = randn(T, 100)
xtrue = A\b

# FISTA
niter = 20
x0 = randn(T, 100)
L = spectral_radius(A'*A, randn(T,100); niter=1000)
prox(x, λ) = x
# Nesterov = true
Nesterov = false
opt_fista = FISTA(L, prox; Nesterov=Nesterov)
fval_fista = Array{T,1}(undef, niter)
x = deepcopy(x0)
for i = 1:niter
    r = A*x-b
    fval_fista[i] = norm(r)^2/2
    g = A'*r
    update!(opt_fista, x, g)
end
err = norm(x-xtrue)/norm(xtrue)