using BlindMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Experiment name/type
phantom = "QuasiInvivo3D_256"; motion = "sudden_motion"
experiment_name = string(phantom, "/", motion)

# Setting folder/savefiles
phantom_folder = string(pwd(), "/data/", phantom, "/")
data_folder    = string(pwd(), "/data/", experiment_name, "/")
results_folder = string(pwd(), "/data/", experiment_name, "/results/")
figures_folder = string(pwd(), "/data/", experiment_name, "/figures/")

# Loading data
ground_truth = load(string(phantom_folder, "ground_truth.jld"))["ground_truth"]
θ_true = load(string(data_folder, "motion_ground_truth.jld"))["θ"]
prior = load(string(phantom_folder, "prior.jld"))["prior"]
X = load(string(data_folder, "data.jld"))["X"]; h = tuple(X.h...)
K = load(string(data_folder, "data.jld"))["K"]
data = load(string(data_folder, "data.jld"))["data"]

# Setting Fourier operator
F = nfft(X, K; tol=1f-5)
nt, nk = size(K)
# u_conventional = F(zeros(Float32,nt,6))'*data

# Parameter estimation options
nt, _ = size(F.K)
ti = Float32.(1:nt)
# ti = Float32.(range(1, nt; length=2^7+1))
t  = Float32.(1:nt)
Ip = interpolation1d_motionpars_linop(ti, t)
D = derivative1d_motionpars_linop(t, 2; pars=(true, true, true, true, true, true))/4f0
opt_parest = parameter_estimation_options(; niter=10, steplength=1f0, λ=0f0, cdiag=1f-5, cid=1f-1, reg_matrix=D, interp_matrix=Ip, W=nothing, verbose=true)

# Parameter estimation
θ0 = zeros(Float32, length(ti)*6)
θ, fval = parameter_estimation(F, ground_truth, data, θ0, opt_parest)
θ = reshape(Ip*θ, nt, 6)

# Image reconstruction options
h = tuple(X.h...)
η = 1f0*structural_mean(prior; h=h)
P = structural_weight(prior; η=η, h=h)
# P = nothing
g = gradient_norm(2, 1, size(prior), h; weight=P, T=ComplexF32)
ε = 0.5f0*g(ground_truth)
# ε = 0.8f0*g(u_conventional)
opt_proj = opt_fista(1f0/12f0; niter=10, Nesterov=true)
prox(u, _) = project(u, ε, g, opt_proj)
opt_recon = image_reconstruction_FISTA_options(; niter=10, steplength=nothing, niter_step_est=10, prox=prox, Nesterov=true, verbose=true)

# Image reconstruction
u0 = zeros(ComplexF32, X.n)
u, fval = image_reconstruction(F(θ_true), data, u0, opt_recon)