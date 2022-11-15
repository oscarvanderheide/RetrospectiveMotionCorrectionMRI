using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, UtilitiesForMRI,  FastSolversForWeightedTV, PyPlot, JLD

# Folders
experiment_name = "26-08-2022-volunteer-study"
exp_folder = string(pwd(), "/data/", experiment_name, "/")
unprocessed_scans_folder = string(exp_folder, "unprocessed_scans/")
data_folder = string(exp_folder, "data/")
~isdir(data_folder) && mkdir(data_folder)
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)

# Loop over experiment
for experiment_subname = ["vol1_priorT1","vol2_priorT1"]

    # Setting files
    @info experiment_subname
    data_file = string("data_", experiment_subname, ".jld")
    unprocessed_scans_file = string("unprocessed_scans_", experiment_subname, ".jld")

    # Loading unprocessed data
    prior = load(string(unprocessed_scans_folder, unprocessed_scans_file))["prior"]
    ground_truth = load(string(unprocessed_scans_folder, unprocessed_scans_file))["ground_truth"]
    corrupted_motion1 = load(string(unprocessed_scans_folder, unprocessed_scans_file))["corrupted_motion1"]
    corrupted_motion2 = load(string(unprocessed_scans_folder, unprocessed_scans_file))["corrupted_motion2"]
    corrupted_motion3 = load(string(unprocessed_scans_folder, unprocessed_scans_file))["corrupted_motion3"]
    mask = load(string(unprocessed_scans_folder, unprocessed_scans_file))["mask"]
    fov = load(string(unprocessed_scans_folder, unprocessed_scans_file))["fov"]
    permutation_dims = load(string(unprocessed_scans_folder, unprocessed_scans_file))["permutation_dims"]
    idx_phase_encoding = load(string(unprocessed_scans_folder, unprocessed_scans_file))["idx_phase_encoding"]
    idx_readout = load(string(unprocessed_scans_folder, unprocessed_scans_file))["idx_readout"]

    # Denoise prior
    X = spatial_geometry(fov, size(prior)); h = spacing(X)
    opt = FISTA_options(4f0*sum(1f0./h.^2); Nesterov=true, niter=20)
    g = gradient_norm(2, 1, size(prior), h; complex=true, options=opt)
    prior = proj(prior, 0.8f0*g(prior), g)

    # Generating synthetic data
    K = kspace_sampling(X, permutation_dims[1:2]; phase_encode_sampling=idx_phase_encoding, readout_sampling=idx_readout)
    F = nfft_linop(X, K)
    data_motion1 = F*corrupted_motion1
    data_motion2 = F*corrupted_motion2
    data_motion3 = F*corrupted_motion3

    # Saving data
    orientation = Orientation((2,1,3), (true,false,true))
    save(string(data_folder, data_file), "data_motion1", data_motion1, "data_motion2", data_motion2, "data_motion3", data_motion3, "X", X, "K", K, "ground_truth", ground_truth, "prior", prior, "corrupted_motion1", corrupted_motion1, "corrupted_motion2", corrupted_motion2, "corrupted_motion3", corrupted_motion3, "orientation", orientation, "mask", mask)

    # Plotting
    figures_subfolder = string(figures_folder, experiment_subname, "/")
    ~isdir(figures_subfolder) && mkdir(figures_subfolder)
    plot_volume_slices(abs.(ground_truth); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_subfolder, "ground_truth.png"), orientation=orientation)
    plot_volume_slices(abs.(prior); spatial_geometry=X, vmin=0, vmax=norm(prior, Inf), savefile=string(figures_subfolder, "prior.png"), orientation=orientation)
    plot_volume_slices(abs.(corrupted_motion1); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_subfolder, "corrupted_motion1.png"), orientation=orientation)
    plot_volume_slices(abs.(corrupted_motion2); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_subfolder, "corrupted_motion2.png"), orientation=orientation)
    plot_volume_slices(abs.(corrupted_motion3); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_subfolder, "corrupted_motion3.png"), orientation=orientation)
    close("all")

end