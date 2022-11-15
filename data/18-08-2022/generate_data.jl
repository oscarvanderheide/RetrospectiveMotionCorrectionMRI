using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, UtilitiesForMRI,  FastSolversForWeightedTV, PyPlot, JLD

# Setting folder/savefiles
experiment_name = "18-08-2022"
data_folder = string(pwd(), "/data/", experiment_name, "/")

# Loading unprocessed data
unprocessed_file = "unprocessed_scans.jld"
fov = load(string(data_folder, unprocessed_file))["fov"]
prior = load(string(data_folder, unprocessed_file))["FLAIR_nomotion"]
T1_motion = load(string(data_folder, unprocessed_file))["T1_motion"]
ground_truth = load(string(data_folder, unprocessed_file))["T1_nomotion"]
permutation_dims = load(string(data_folder, unprocessed_file))["permutation_dims"]
coord_phase_encoding = load(string(data_folder, unprocessed_file))["coord_phase_encoding"]
coord_readout = load(string(data_folder, unprocessed_file))["coord_readout"]
# permutation_dims = (3, 2, 1)

# Denoise prior
X = spatial_geometry(fov, size(T1_motion)); h = spacing(X);
opt = FISTA_optimizer(4f0*sum(1 ./h.^2); Nesterov=true, niter=20)
g = gradient_norm(2, 1, size(prior), spacing(X), opt; complex=true)
prior = proj(prior, 0.8f0*g(prior), g)
prior ./= norm(prior, Inf)

# Save ground-truth and prior
save(string(data_folder, "/ground_truth.jld"), "ground_truth", ground_truth)
save(string(data_folder, "/prior.jld"), "prior", prior)

# Generating synthetic data
K = kspace_sampling(permutation_dims, coord_phase_encoding, coord_readout)
F = nfft_linop(X, K)
data = F*T1_motion

# Saving data
save(string(data_folder, "data.jld"), "data", data, "X", X, "K", K)

# Plotting
figures_folder = string(pwd(), "/data/", experiment_name, "/figures/")
~isdir(figures_folder) && mkdir(figures_folder)
vmin = 0f0; vmax = maximum(abs.(ground_truth))
x, y, z = div.(size(ground_truth), 2).+1
plot_slices = (VolumeSlice(1, x), VolumeSlice(2, y), VolumeSlice(3, z))
plot_volume_slices(abs.(ground_truth); slices=plot_slices, spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "ground_truth.png"), orientation=orientation)
plot_volume_slices(abs.(T1_motion); slices=plot_slices, spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "conventional.png"), orientation=orientation)
plot_volume_slices(abs.(prior); slices=plot_slices, spatial_geometry=X, vmin=0, vmax=norm(prior, Inf), savefile=string(figures_folder, "prior.png"), orientation=orientation)