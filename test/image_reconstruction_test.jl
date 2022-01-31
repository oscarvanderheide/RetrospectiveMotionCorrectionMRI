using LinearAlgebra, BlindMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, Flux, ImageFiltering, PyPlot, JLD

# Numerical phantom
h = [1f0, 1f0, 1f0]
u_true = Float32.(load("./data/shepplogan3D_128.jld")["u"])
n = size(u_true)
u_true = complex(u_true/norm(u_true, Inf))

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
ssim_approx = ssim(u_approx, u_true)
psnr_approx = psnr(u_approx, u_true)
println("Approx.: SSIM = ", ssim_approx, ", PSNR = ", psnr_approx)

# Optimization options
niter = 10
λ = 1f-6
steplength = 1f0
reg_mean = zeros(ComplexF32, n)
hist_size = 10
β = 1f0
opt = image_reconstruction_options(; niter=niter, steplength=steplength, λ=λ, hist_size=hist_size, β=β, reg_mean=reg_mean, verbose=true)

# Solution
u0 = zeros(ComplexF32, n)
u_sol, fval = image_reconstruction(Fθ, d, u0, opt)
ssim_sol = ssim(u_sol, u_true)
psnr_sol = psnr(u_sol, u_true)
println("Recon. sol: SSIM = ", ssim_sol, ", PSNR = ", psnr_sol)