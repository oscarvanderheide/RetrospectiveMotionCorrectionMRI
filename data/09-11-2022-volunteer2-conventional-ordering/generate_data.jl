using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, UtilitiesForMRI,  FastSolversForWeightedTV, PyPlot, JLD

# Folders & files
experiment_name = "09-11-2022-volunteer2-conventional-ordering"; @info experiment_name
exp_folder = string(pwd(), "/data/", experiment_name, "/")
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)
data_file = "data.jld"
unprocessed_scans_file = "unprocessed_scans.jld"

# Loading unprocessed data
prior, prior_reg, mask_prior, ground_truth, mask_ground_truth, corrupted_motion1, corrupted_motion2, corrupted_motion3, fov, permutation_dims, idx_phase_encoding, idx_readout = load(string(exp_folder, unprocessed_scans_file), "prior", "prior_reg", "mask_prior", "ground_truth", "mask_ground_truth", "corrupted_motion1", "corrupted_motion2", "corrupted_motion3", "fov", "permutation_dims", "idx_phase_encoding", "idx_readout")

# Denoise prior
X = spatial_geometry(fov, size(prior)); h = spacing(X)
opt = FISTA_options(4f0*sum(1f0./h.^2); Nesterov=true, niter=20)
g = gradient_norm(2, 1, size(prior), h; complex=true, options=opt)
prior = proj(prior, 0.5f0*g(prior), g)

# Generating synthetic data
K = kspace_sampling(X, permutation_dims[1:2]; phase_encode_sampling=idx_phase_encoding, readout_sampling=idx_readout)
F = nfft_linop(X, K)
data_motion1 = F*corrupted_motion1
data_motion2 = F*corrupted_motion2
data_motion3 = F*corrupted_motion3

# Rigid registration
opt_reg = rigid_registration_options(; niter=20, verbose=true)
corrupted_motion1_reg, _ = rigid_registration(corrupted_motion1, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=5)
corrupted_motion2_reg, _ = rigid_registration(corrupted_motion2, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=5)
corrupted_motion3_reg, _ = rigid_registration(corrupted_motion3, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=5)

# Plotting
orientation = Orientation((2,1,3), (true,false,true))
plot_volume_slices(abs.(ground_truth); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "ground_truth.png"), orientation=orientation)
plot_volume_slices(abs.(prior); spatial_geometry=X, vmin=0, vmax=norm(prior_reg, Inf), savefile=string(figures_folder, "prior.png"), orientation=orientation)
plot_volume_slices(abs.(prior_reg); spatial_geometry=X, vmin=0, vmax=norm(prior_reg, Inf), savefile=string(figures_folder, "prior_reg.png"), orientation=orientation)
plot_volume_slices(abs.(corrupted_motion1_reg); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "corrupted_motion1.png"), orientation=orientation)
plot_volume_slices(abs.(corrupted_motion2_reg); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "corrupted_motion2.png"), orientation=orientation)
plot_volume_slices(abs.(corrupted_motion3_reg); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_folder, "corrupted_motion3.png"), orientation=orientation)
close("all")

# Saving data
save(string(exp_folder, data_file), "data_motion1", data_motion1, "data_motion2", data_motion2, "data_motion3", data_motion3, "X", X, "K", K, "ground_truth", ground_truth, "prior", prior, "prior_reg", prior_reg, "corrupted_motion1", corrupted_motion1, "corrupted_motion2", corrupted_motion2, "corrupted_motion3", corrupted_motion3, "corrupted_motion1_reg", corrupted_motion1_reg, "corrupted_motion2_reg", corrupted_motion2_reg, "corrupted_motion3_reg", corrupted_motion3_reg, "orientation", orientation, "mask_prior", mask_prior, "mask_ground_truth", mask_ground_truth)