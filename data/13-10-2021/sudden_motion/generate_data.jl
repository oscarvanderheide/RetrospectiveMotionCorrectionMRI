using LinearAlgebra, BlindMotionCorrectionMRI, UtilitiesForMRI, PyPlot, JLD, Statistics

# Setting folder/savefiles
phantom_id = "Invivo3D_13-10-2021"
motion_id  = "sudden_motion"
data_folder = string(pwd(), "/data/", phantom_id, "/")

# Loading data
h = tuple(Float32.(load(string(data_folder, "unprocessed_scans.jld"))["spacing"])...)
T2_motion    = ComplexF32.(load(string(data_folder, "unprocessed_scans.jld"))["T2_motion"])
ground_truth = ComplexF32.(load(string(data_folder, "unprocessed_scans.jld"))["T2_nomotion"])
# prior        = ComplexF32.(abs.(load(string(data_folder, "unprocessed_scans.jld"))["T1_nomotion"]))

# # Denoise prior
# g = gradient_norm(2, 1, size(prior), (1f0,1f0,1f0); T=ComplexF32)
# ε = 0.8f0*g(prior)
# opt_proj = opt_fista(1f0/12f0; niter=20, Nesterov=true)
# prior = proj(prior, ε, g, opt_proj)

# Normalization
# prior = prior/norm(prior, Inf)
# c = norm(ground_truth)
# ground_truth ./= c
# T2_motion ./= c
T2_motion ./= norm(T2_motion)
ground_truth .*= dot(ground_truth, T2_motion)/norm(ground_truth)^2

# Save ground-truth and prior
save(string(data_folder, "/ground_truth.jld"), "ground_truth", ground_truth)
save(string(data_folder, "/prior.jld"), "prior", prior)

# Generating synthetic data
n = size(T2_motion)
o = (81f0, 11f0, 51f0).-(div.(n,2).+1)
X = spatial_sampling(Float32, n; h=h, o=o)
K = kspace_Cartesian_sampling(X; phase_encoding=(1,2))
F = nfft_linop(X, K; tol=1f-6)
data = F*T2_motion

# Saving data
save(string(data_folder, motion_id, "/data.jld"), "data", data, "X", X, "K", K)