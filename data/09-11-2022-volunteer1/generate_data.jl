using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, UtilitiesForMRI,  FastSolversForWeightedTV, PyPlot, JLD

# Folders & files
experiment_name = "09-11-2022-volunteer1"; @info experiment_name
exp_folder = string(pwd(), "/data/", experiment_name, "/")
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)
data_file = "data.jld"
unprocessed_scans_file = "unprocessed_scans.jld"

# Loading unprocessed data
prior = load(string(exp_folder, unprocessed_scans_file))["prior"]
ground_truth = load(string(exp_folder, unprocessed_scans_file))["ground_truth"]
corrupted_motion1 = load(string(exp_folder, unprocessed_scans_file))["corrupted1"]
corrupted_motion2 = load(string(exp_folder, unprocessed_scans_file))["corrupted2"]
corrupted_motion3 = load(string(exp_folder, unprocessed_scans_file))["corrupted3"]
mask_prior = load(string(exp_folder, unprocessed_scans_file))["mask_prior"]
mask_ground_truth = load(string(exp_folder, unprocessed_scans_file))["mask_ground_truth"]
fov = load(string(exp_folder, unprocessed_scans_file))["fov"]
permutation_dims = load(string(exp_folder, unprocessed_scans_file))["permutation_dims"]
idx_phase_encoding = load(string(exp_folder, unprocessed_scans_file))["idx_phase_encoding"]
idx_readout = load(string(exp_folder, unprocessed_scans_file))["idx_readout"]

# Denoise prior
X = spatial_geometry(fov, size(prior)); h = spacing(X)
opt = FISTA_options(4f0*sum(1f0./h.^2); Nesterov=true, niter=20)
g = gradient_norm(2, 1, size(prior), h; complex=true, options=opt)
prior = proj(prior, 0.5f0*g(prior), g)
z = (sin.(range(-Float32(pi)/2, Float32(pi)/2; length=size(prior,3)-130)).+1)./2
prior[:,:,end-length(z)+1:end] .*= reshape(z[end:-1:1], 1, 1, :)

# Generating synthetic data
K = kspace_sampling(X, permutation_dims[1:2]; phase_encode_sampling=idx_phase_encoding, readout_sampling=idx_readout)
F = nfft_linop(X, K)
data_motion1 = F*corrupted_motion1
data_motion2 = F*corrupted_motion2
data_motion3 = F*corrupted_motion3

# Saving data
orientation = Orientation((2,1,3), (true,false,true))
save(string(exp_folder, data_file), "data_motion1", data_motion1, "data_motion2", data_motion2, "data_motion3", data_motion3, "X", X, "K", K, "ground_truth", ground_truth, "prior", prior, "corrupted_motion1", corrupted_motion1, "corrupted_motion2", corrupted_motion2, "corrupted_motion3", corrupted_motion3, "orientation", orientation, "mask_prior", mask_prior, "mask_ground_truth", mask_ground_truth)

# Plotting
plot_volume_slices(abs.(ground_truth); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "ground_truth.png"), orientation=orientation)
plot_volume_slices(abs.(prior); spatial_geometry=X, vmin=0, vmax=norm(prior, Inf), savefile=string(figures_folder, "prior.png"), orientation=orientation)
plot_volume_slices(abs.(corrupted_motion1); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "corrupted_motion1.png"), orientation=orientation)
plot_volume_slices(abs.(corrupted_motion2); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "corrupted_motion2.png"), orientation=orientation)
plot_volume_slices(abs.(corrupted_motion3); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "corrupted_motion3.png"), orientation=orientation)
close("all")