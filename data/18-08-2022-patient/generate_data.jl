using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, UtilitiesForMRI,  FastSolversForWeightedTV, PyPlot, JLD

# Folders & files
experiment_name = "18-08-2022-patient"
exp_folder = string(pwd(), "/data/", experiment_name, "/")
unprocessed_scans_folder = exp_folder
data_folder = string(exp_folder, "data/")
~isdir(data_folder) && mkdir(data_folder)
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)
data_file = "data.jld"
unprocessed_scans_file = "unprocessed_scans.jld"

# Loading unprocessed data
prior = abs.(load(string(unprocessed_scans_folder, unprocessed_scans_file))["prior"]).+0*im; prior ./= norm(prior, Inf)
ground_truth = load(string(unprocessed_scans_folder, unprocessed_scans_file))["ground_truth"]
corrupted = load(string(unprocessed_scans_folder, unprocessed_scans_file))["corrupted"]
fov = load(string(unprocessed_scans_folder, unprocessed_scans_file))["fov"]
permutation_dims = load(string(unprocessed_scans_folder, unprocessed_scans_file))["permutation_dims"]
coord_phase_encoding = load(string(unprocessed_scans_folder, unprocessed_scans_file))["coord_phase_encoding"]
coord_readout = load(string(unprocessed_scans_folder, unprocessed_scans_file))["coord_readout"]

# Denoise prior
X = spatial_geometry(fov, size(corrupted)); h = spacing(X)
opt = FISTA_optimizer(4f0*sum(1 ./h.^2); Nesterov=true, niter=20)
g = gradient_norm(2, 1, size(prior), h, opt; complex=true)
prior = project(prior, 0.5f0*g(prior), g)
prior ./= norm(prior, Inf)

# Generating synthetic data
K = kspace_sampling(permutation_dims, coord_phase_encoding, coord_readout)
F = nfft_linop(X, K)
data = F*corrupted

# Saving data
orientation = Orientation((2,1,3), (true,false,true))
save(string(data_folder, data_file), "data", data, "X", X, "K", K, "ground_truth", ground_truth, "prior", prior, "corrupted", corrupted, "orientation", orientation)

# Plotting
vmin = 0f0; vmax = maximum(abs.(ground_truth))
plot_volume_slices(abs.(ground_truth); spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "ground_truth.png"), orientation=orientation)
plot_volume_slices(abs.(corrupted); spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "conventional.png"), orientation=orientation)
plot_volume_slices(abs.(prior); spatial_geometry=X, vmin=0, vmax=norm(prior, Inf), savefile=string(figures_folder, "prior.png"), orientation=orientation)
close("all")