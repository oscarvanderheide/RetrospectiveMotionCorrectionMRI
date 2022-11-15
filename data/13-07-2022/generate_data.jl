using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, UtilitiesForMRI,  FastSolversForWeightedTV, PyPlot, JLD

# Folders
experiment_name = "13-07-2022"
exp_folder = string(pwd(), "/data/", experiment_name, "/")
unprocessed_scans_folder = string(exp_folder, "unprocessed_scans/")
data_folder = string(exp_folder, "data/")
~isdir(data_folder) && mkdir(data_folder)
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)

for recon_type = ["custom", "DICOM"]

    # Setting files
    data_file = string("data_", recon_type, ".jld")
    unprocessed_scans_file = string("unprocessed_scans_", recon_type, ".jld")

    # Loading unprocessed data
    prior = load(string(unprocessed_scans_folder, unprocessed_scans_file))["prior"]; prior ./= norm(prior, Inf)
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
    prior = proj(prior, 0.8f0*g(prior), g)
    prior ./= norm(prior, Inf)

    # Generating synthetic data
    K = kspace_sampling(permutation_dims, coord_phase_encoding, coord_readout)
    F = nfft_linop(X, K)
    data = F*corrupted

    # Saving data
    orientation = Orientation((2,1,3), (true,false,true))
    save(string(data_folder, data_file), "data", data, "X", X, "K", K, "ground_truth", ground_truth, "prior", prior, "corrupted", corrupted, "orientation", orientation)

    # Plotting
    figures_subfolder = string(figures_folder, recon_type, "/")
    ~isdir(figures_subfolder) && mkdir(figures_subfolder)
    vmin = 0f0; vmax = maximum(abs.(ground_truth))
    x, y, z = div.(size(ground_truth), 2).+1
    plot_slices = (VolumeSlice(1, x), VolumeSlice(2, y), VolumeSlice(3, z))
    plot_volume_slices(abs.(ground_truth); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_subfolder, "ground_truth.png"), orientation=orientation)
    plot_volume_slices(abs.(corrupted); spatial_geometry=X, vmin=0, vmax=norm(ground_truth, Inf), savefile=string(figures_subfolder, "corrupted.png"), orientation=orientation)
    plot_volume_slices(abs.(prior); spatial_geometry=X, vmin=0, vmax=norm(prior, Inf), savefile=string(figures_subfolder, "prior.png"), orientation=orientation)
    close("all")

end