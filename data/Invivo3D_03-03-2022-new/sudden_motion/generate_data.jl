using LinearAlgebra, BlindMotionCorrectionMRI, UtilitiesForMRI,  FastSolversForWeightedTV, PyPlot, JLD, Statistics

# Setting folder/savefiles
phantom_id = "Invivo3D_03-03-2022-new"; motion_id  = "sudden_motion"
data_folder = string(pwd(), "/data/", phantom_id, "/", motion_id, "/")

# Loading data
h = (0.9583f0, 0.9583f0, 0.6f0)
# h = (1f0, 1f0, 1f0)
FLAIR_motion = ComplexF32.(load(string(data_folder, "unprocessed_scans.jld"))["FLAIR_motion"])
ground_truth = ComplexF32.(load(string(data_folder, "unprocessed_scans.jld"))["FLAIR_nomotion"])

# # Rigid registration
# opt_reg = rigid_registration_options(Float32; steplength=1f0, niter=50, verbose=true)
# prior, _, _ = rigid_registration(prior/norm(prior, Inf), complex(abs.(FLAIR_motion))/norm(FLAIR_motion, Inf), nothing, opt_reg)

# # Denoise prior
# g = gradient_norm(2, 1, size(prior), (1f0, 1f0, 1f0); T=ComplexF32)
# opt_proj = opt_fista(1f0/12f0; niter=20, Nesterov=true)
# prior = project(prior, 0.5f0*g(prior), g, opt_proj)
# prior ./= norm(prior, Inf)

# Normalization
# c = norm(ground_truth)
# ground_truth ./= c
# FLAIR_motion ./= c
# FLAIR_motion ./= norm(FLAIR_motion)
# ground_truth .*= dot(ground_truth, FLAIR_motion)/norm(ground_truth)^2
# ground_truth .*= mean(FLAIR_motion)/mean(ground_truth)
# ground_truth ./= norm(ground_truth)

# Save ground-truth and prior
save(string(data_folder, "/ground_truth.jld"), "ground_truth", ground_truth)
# save(string(data_folder, "/prior.jld"), "prior", prior)

# Generating synthetic data
n = size(FLAIR_motion)
X = spatial_sampling(Float32, n; h=h)
K = kspace_Cartesian_sampling(X; phase_encoding=(2,3))
F = nfft_linop(X, K; tol=1f-6)
data = F*FLAIR_motion

# Saving data
save(string(data_folder, "data.jld"), "data", data, "X", X, "K", K)