using LinearAlgebra, RetrospectiveMotionCorrectionMRI, ConvexOptimizationUtils, UtilitiesForMRI,  FastSolversForWeightedTV, PyPlot, JLD

# Setting folder/savefiles
experiment_name = "17-06-2022_SENSE"
data_folder = string(pwd(), "/data/", experiment_name, "/")

# Loading unprocessed data
unprocessed_file = "unprocessed_scans.jld"
fov = load(string(data_folder, unprocessed_file))["fov"]
prior = load(string(data_folder, unprocessed_file))["T1"]
T2_motion = load(string(data_folder, unprocessed_file))["T2_motion"]
ground_truth = load(string(data_folder, unprocessed_file))["T2_nomotion"]

# Denoise prior
X = spatial_geometry(fov, size(T2_motion)); h = spacing(X)
opt = FISTA_optimizer(4f0*sum(1 ./h.^2); Nesterov=true, niter=20)
g = gradient_norm(2, 1, size(prior), spacing(X), opt; complex=true)
prior = project(prior, 0.5f0*g(prior), g)
prior ./= norm(prior, Inf)

# Save ground-truth and prior
save(string(data_folder, "/ground_truth.jld"), "ground_truth", ground_truth)
save(string(data_folder, "/prior.jld"), "prior", prior)

# Generating synthetic data
phase_encoding = (3,1)
K = kspace_sampling(X, phase_encoding)
F = nfft_linop(X, K)
data = F*T2_motion

# Saving data
save(string(data_folder, "data.jld"), "data", data, "X", X, "K", K)