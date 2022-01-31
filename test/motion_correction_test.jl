using LinearAlgebra, BlindMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, Flux, PyPlot, JLD

# Numerical phantom
h = [1f0, 1f0, 1f0]
u_true = Float32.(load("./data/shepplogan3D_128.jld")["u"])
n = size(u_true)
u_true = complex(u_true/norm(u_true, Inf))

# Fourier operator
X = spatial_sampling(n; h=h)
K = kspace_sampling(X; readout=:z, phase_encode=:xy); nt, nk = size(K)
F = nfft(X, K; tol=1f-5)
θ_true = zeros(Float32, n[1:2]..., 6)
θ_true[:, 65+3end, 1:3] .= 3f0
θ_true[:, 65+3end, 4:6] .= pi/180*3
θ_true = reshape(θ_true, prod(n[1:2]), 6)
Fθ_true = F(θ_true)

# Data
snr = 90f0 # dB
d = Fθ_true*u_true
d .+= 10^(-snr/20f0)*norm(d)*randn(ComplexF32, size(d))

# Approximated solution
u_approx = nfft_linop(X, K; tol=1f-5)'*d
ssim_approx = ssim(u_approx, u_true)
psnr_approx = psnr(u_approx, u_true)
println("Approx.: SSIM = ", ssim_approx, ", PSNR = ", psnr_approx)

# Optimization options
## Image reconstruction
opt_recon = image_reconstruction_options(; niter=10, steplength=1f0, λ=1f0, hist_size=10, β=1f0, verbose=true)
ti = Float32.(range(1, nt; length=Int64(nt/1)))
t = Float32.(1:nt)
Ip = interpolation1d_motionpars_linop(ti, t)
D = derivative1d_motionpars_linop(t, 2; pars=(true, true, true, true, true, true))/4f0
## Parameter estimation
opt_parest = parameter_estimation_options(; niter=1, steplength=1f0, λ=0f0, cdiag=1f-5, cid=1f-1, reg_matrix=D, interp_matrix=Ip, verbose=true)
## Denoiser
g = gradient_norm(2, 1, n, tuple(h...); T=ComplexF32, weight=nothing)
ε = 0.8f0*g(u_true)
opt_denoiser = opt_fista(1f0/12f0; niter=10, Nesterov=true)
denoiser(u) = project(u, ε, g, opt_denoiser)
## Overall options
opt = motion_correction_options(; niter=100, denoiser=denoiser, image_reconstruction_options=opt_recon, parameter_estimation_options=opt_parest, verbose=true) 

# Solution
u = zeros(ComplexF32, n)
θ = zeros(Float32, size(Ip, 2))
u_sol, θ_true_sol, fval = motion_corrected_reconstruction(F, d, u, θ, opt)
ssim_sol = ssim(u_sol, u_true)
psnr_sol = psnr(u_sol, u_true)
println("Recon. sol: SSIM = ", ssim_sol, ", PSNR = ", psnr_sol)