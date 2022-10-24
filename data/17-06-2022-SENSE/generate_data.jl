using LinearAlgebra, RetrospectiveMotionCorrectionMRI, ConvexOptimizationUtils, UtilitiesForMRI,  FastSolversForWeightedTV, PyPlot, JLD

# Folders
experiment_name = "17-06-2022-SENSE"; @info experiment_name
exp_folder = string(pwd(), "/data/", experiment_name, "/")
unprocessed_scans_folder = string(exp_folder, "unprocessed_scans/")
data_folder = string(exp_folder, "data/")
~isdir(data_folder) && mkdir(data_folder)
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)

# Setting files
data_file = "data.jld"
unprocessed_scans_file = "unprocessed_scans.jld"

# Loading unprocessed data
prior = load(string(unprocessed_scans_folder, unprocessed_scans_file))["prior"]; prior ./= norm(prior, Inf)
ground_truth = load(string(unprocessed_scans_folder, unprocessed_scans_file))["ground_truth"]
corrupted = load(string(unprocessed_scans_folder, unprocessed_scans_file))["corrupted"]
fov = load(string(unprocessed_scans_folder, unprocessed_scans_file))["fov"]
permutation_dims = load(string(unprocessed_scans_folder, unprocessed_scans_file))["permutation_dims"]
idx_phase_encoding = load(string(unprocessed_scans_folder, unprocessed_scans_file))["idx_phase_encoding"]
idx_readout = load(string(unprocessed_scans_folder, unprocessed_scans_file))["idx_readout"]

# Denoise prior
X = spatial_geometry(fov, size(corrupted)); h = spacing(X)
opt = FISTA_optimizer(4f0*sum(1 ./h.^2); Nesterov=true, niter=20)
g = gradient_norm(2, 1, size(prior), h, opt; complex=true)
prior = project(prior, 0.8f0*g(prior), g)
prior ./= norm(prior, Inf)

# Generating synthetic data
K = kspace_sampling(X, permutation_dims[1:2]; phase_encode_sampling=idx_phase_encoding, readout_sampling=idx_readout)
F = nfft_linop(X, K)
data = F*corrupted

# Saving data
orientation = Orientation((1,2,3), (false, false, false))
save(string(data_folder, data_file), "data", data, "X", X, "K", K, "ground_truth", ground_truth, "prior", prior, "corrupted", corrupted, "orientation", orientation)

# Plotting
plot_volume_slices(abs.(ground_truth); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "ground_truth.png"), orientation=orientation)
plot_volume_slices(abs.(corrupted); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "corrupted.png"), orientation=orientation)
plot_volume_slices(abs.(prior); spatial_geometry=X, vmin=0, vmax=norm(prior, Inf), savefile=string(figures_folder, "prior.png"), orientation=orientation)
close("all")