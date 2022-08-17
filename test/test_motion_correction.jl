using RetrospectiveMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, ConvexOptimizationUtils, LinearAlgebra, Test

# Spatial geometry
fov = (1f0, 2f0, 2f0)
n = (64, 64, 64)
o = (0.5f0, 1f0, 1f0)
X = spatial_geometry(fov, n; origin=o)

# Cartesian sampling in k-space
phase_encoding = (1,2)
K = kspace_sampling(X, phase_encoding)
nt, nk = size(K)

# Fourier operator
F = nfft_linop(X, K)

# Setting ground-truth and data
ground_truth = zeros(ComplexF32, n); ground_truth[div(64,2)+1-10:div(64,2)+1+10, div(64,2)+1-10:div(64,2)+1+10, div(64,2)+1-10:div(64,2)+1+10] .= 1
nt, _ = size(K)
θ_true = zeros(Float32, nt, 6)
θ_true[div(nt,2)+51:end,:] .= reshape([0.0f0, 0.0f0, 0.0f0, Float32(pi)/180*10, 0f0, 0f0], 1, 6)
d = F(θ_true)*ground_truth
u_conventional = F'*d

# Image reconstruction options
h = spacing(X); LD = 4f0*sum(1 ./h.^2)
opt_inner = FISTA_optimizer(LD; Nesterov=true, niter=10)
g = gradient_norm(2, 1, size(ground_truth), h, opt_inner; complex=true)
ε = 0.8f0*g(ground_truth)
h = indicator(g ≤ ε)
opt_imrecon = image_reconstruction_options(; prox=h, Lipschitz_constant=1f0, Nesterov=true, niter=5, verbose=true, fun_history=true)

# Parameter estimation options
ti = Float32.(range(1, nt; length=16))
# ti = Float32.(1:nt)
t = Float32.(1:nt)
Ip = interpolation1d_motionpars_linop(ti, t)
D = derivative1d_motionpars_linop(t, 2; pars=(true, true, true, true, true, true))/4f0
opt_parest = parameter_estimation_options(; niter=5, steplength=1f0, λ=0f0, cdiag=1f-5, cid=1f-1, reg_matrix=D, interp_matrix=Ip, verbose=true, fun_history=true)

# Solution
opt = motion_correction_options(; image_reconstruction_options=opt_imrecon, parameter_estimation_options=opt_parest, niter=40, niter_estimate_Lipschitz=3, verbose=true, fun_history=true)
θ0 = zeros(Float32, size(Ip,2))
u0 = zeros(ComplexF32, n)
u, θ = motion_corrected_reconstruction(F, d, u0, θ0, opt)
θ = reshape(Ip*vec(θ), length(t), 6)

# Comparison
u_conventional = F'*d
@info string("Conventional: (psnr,ssim)=(", psnr(u_conventional, ground_truth; preproc=x->abs.(x)), ",", ssim(u_conventional, ground_truth; preproc=x->abs.(x)),")")
@info string("Motion-corrected: (psnr,ssim)=(", psnr(u, ground_truth; preproc=x->abs.(x)), ",", ssim(u, ground_truth; preproc=x->abs.(x)))