using BlindMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, AbstractLinearOperators, PyPlot, JLD, ImageFiltering, LinearAlgebra
include(string(pwd(), "/scripts/plot_results.jl"))

# Experiment name/type
phantom = "Invivo3D_03-03-2022"; motion = "sudden_motion"
experiment_name = string(phantom, "/", motion)

# Setting folder/savefiles
phantom_folder = string(pwd(), "/data/", phantom, "/")
data_folder    = string(pwd(), "/data/", experiment_name, "/")

# Loading data
X = load(string(data_folder, "data.jld"))["X"]
K = load(string(data_folder, "data.jld"))["K"]
data = load(string(data_folder, "data.jld"))["data"]
ground_truth = load(string(phantom_folder, "ground_truth.jld"))["ground_truth"]
vmin = 0; vmax = maximum(abs.(ground_truth))

# Structure-guided prior
# struct_prior = true
struct_prior = false
struct_prior && (prior = load(string(phantom_folder, "prior.jld"))["prior"])

# Setting Fourier operator
nt, nk = size(K)
F0 = nfft(X, K; tol=1f-6)(zeros(Float32,nt,6))
u_conventional = F0'*data

# Image reconstruction options
struct_prior ? (η = 1f-2; P = structural_weight(prior; η=η)) : (P = nothing)
g = gradient_norm(2, 1, size(ground_truth), (1f0,1f0,1f0); weight=P, T=ComplexF32)
ε = 0.5f0*g(u_conventional)
opt_proj = opt_fista(1f0/12f0; niter=5, Nesterov=true)
prox(u, _) = project(u, ε, g, opt_proj)
loss = data_residual_loss(ComplexF32, 2, 2)
calibration_options = calibration(:readout, 1f10)
opt_recon = image_reconstruction_FISTA_options(Float32; loss=loss, niter=10, steplength=nothing, niter_EstLipschitzConst=3, prox=prox, Nesterov=true, calibration=calibration_options, verbose=true)

# u0 = zeros(ComplexF32, size(ground_truth))
u0 = u_conventional
u, fval, A = image_reconstruction(F0, data, u0, opt_recon)