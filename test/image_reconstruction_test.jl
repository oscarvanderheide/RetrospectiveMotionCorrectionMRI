using LinearAlgebra, BlindMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, Flux, TestImages, ImageFiltering, PyPlot

# Numerical phantom
n = (128, 128, 128)
h = [1f0, 1f0, 1f0]
u_true = Float32.(TestImages.shepp_logan(n[1:2]...))
u_true = repeat(u_true; outer=(1,1,n[3])); u_true[:,:,1:10] .= 0; u_true[:,:,end-9:end] .= 0
u_true = complex(u_true/norm(u_true, Inf))

# Regularization setup
optTV = opt_fista(1f0/12f0; niter=10, tol_x=1f-5, Nesterov=true)
TV = gradient_norm(2, 1, n, tuple(h...); T=ComplexF32)
ε = 0.5f0*TV(u_true)
prox(u, λ) = project(u, ε, TV, optTV)
# prox(u, λ) = u

# Fourier operator
X = spatial_sampling(n; h=h)
K = kspace_sampling(X; readout=:z, phase_encode=:xy); nt, nk = size(K)
F = nfft(X, K; tol=1f-5)
θ = zeros(Float32, nt, 6)
θ[Int64(floor(0.6*nt)):end, 1:3] .= 3f0
θ[Int64(floor(0.6*nt)):end, 4:6] .= pi/180*3
Fθ = F(θ)

# Data
d = Fθ*u_true

# Optimization options
L = spectral_radius(Fθ'*Fθ, randn(ComplexF32, n); niter=20)
opt = FISTA_reconstruction_options(prox; niter=10, L=L, Nesterov=true, verbose=true)

# Approximated solution
u_approx = nfft_linop(X, K; tol=1f-5)'*d

# Solution
u_sol, fval = FISTA_reconstruction(Fθ, d; opt=opt)
ssim(u_sol, u_true)
psnr(u_sol, u_true)
ssim(u_approx, u_true)
psnr(u_approx, u_true)