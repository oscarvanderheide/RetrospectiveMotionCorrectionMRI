using LinearAlgebra, BlindMotionCorrectionMRI, UtilitiesForMRI,  FastSolversForWeightedTV, PyPlot, JLD, Statistics

# Setting folder/savefiles
phantom_id = "Invivo3D_03-03-2022"
motion_id  = "sudden_motion"
data_folder = string(pwd(), "/data/", phantom_id, "/")

# Loading data
h = (1f0, 1f0, 1f0)
FLAIR_motion = ComplexF32.(load(string(data_folder, "unprocessed_scans.jld"))["FLAIR_motion"])
ground_truth = ComplexF32.(load(string(data_folder, "unprocessed_scans.jld"))["FLAIR"])
# prior        = ComplexF32.(abs.(load(string(data_folder, "unprocessed_scans.jld"))["T1"]))

# # Rigid registration
# opt_reg = rigid_registration_options(Float32; steplength=1f0, niter=50, verbose=true)
# prior, _, _ = rigid_registration(prior/norm(prior, Inf), complex(abs.(FLAIR_motion))/norm(FLAIR_motion, Inf), nothing, opt_reg)

# # Denoise prior
# g = gradient_norm(2, 1, size(prior), (1f0, 1f0, 1f0); T=ComplexF32)
# opt_proj = opt_fista(1f0/12f0; niter=20, Nesterov=true)
# prior = project(prior, 0.5f0*g(prior), g, opt_proj)
# prior ./= norm(prior, Inf)

# Normalization
c = norm(FLAIR_motion)
ground_truth ./= c
FLAIR_motion ./= c

# # Save ground-truth and prior
save(string(data_folder, "/ground_truth.jld"), "ground_truth", ground_truth)
# save(string(data_folder, "/prior.jld"), "prior", prior)

# Generating synthetic data
n = size(FLAIR_motion)
# o = (121, 17, 91)
# X = spatial_sampling(Float32, n; h=h, o=o)
X = spatial_sampling(Float32, n; h=h)
K = kspace_Cartesian_sampling(X; phase_encoding=(1,3))
F = nfft_linop(X, K; tol=1f-6)
data = F*FLAIR_motion

# Saving data
save(string(data_folder, motion_id, "/data.jld"), "data", data, "X", X, "K", K)