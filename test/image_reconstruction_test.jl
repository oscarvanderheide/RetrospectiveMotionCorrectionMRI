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

# Approximated solution
u_approx = nfft_linop(X, K; tol=1f-5)'*d
println("Approx. sol: SSIM = ", ssim(u_approx, u_true))
println("             PSNR = ", psnr(u_approx, u_true))

# ## FISTA

# # Optimization options
# L = spectral_radius(Fθ'*Fθ, randn(ComplexF32, n); niter=20)
# opt = FISTA_reconstruction_options(prox; niter=10, L=L, Nesterov=true, verbose=true)

# # Solution
# u_sol, fval = FISTA_reconstruction(Fθ, d, opt)
# println("FISTA sol: SSIM = ", ssim(u_sol, u_true))
# println("           PSNR = ", psnr(u_sol, u_true))

## Splitting (regular)

# Optimization options
L = spectral_radius(Fθ'*Fθ, randn(ComplexF32, n); niter=20)
λ = 0.01f0*sqrt(L)
steplength = 1f0/(L+λ^2)
u_ref = zeros(ComplexF32, n)
opt = splitreg_reconstruction_options(; niter=10, steplength=steplength, λ=λ, u_ref=u_ref, verbose=true)

# Solution
u_sol, fval = splitreg_reconstruction(Fθ, d, opt)
println("SplitReg sol: SSIM = ", ssim(u_sol, u_true), ", PSNR = ", psnr(u_sol, u_true))

## Splitting (Anderson)

# Optimization options
λ = 0.01f0*sqrt(L)
steplength = 1f0/(1f0+λ^2)
u_ref = zeros(ComplexF32, n)
hist_size = 10
β = 1f0
opt = splitregAnderson_reconstruction_options(; niter=10, steplength=steplength, λ=λ, u_ref=u_ref, hist_size=10, β=β,verbose=true)

# Solution
u_sol, fval = splitreg_reconstruction(Fθ, d, opt)
println("SplitRegAnderson sol: SSIM = ", ssim(u_sol, u_true), ", PSNR = ", psnr(u_sol, u_true))