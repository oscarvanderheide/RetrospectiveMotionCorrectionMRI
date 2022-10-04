using LinearAlgebra, RetrospectiveMotionCorrectionMRI, ConvexOptimizationUtils, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Folders
experiment_name = "26-08-2022-volunteer-study"; @info string(experiment_name, "\n")
exp_folder = string(pwd(), "/data/", experiment_name, "/")
data_folder = string(exp_folder, "data/")
~isdir(data_folder) && mkdir(data_folder)
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)
results_folder = string(exp_folder, "results/")
~isdir(results_folder) && mkdir(results_folder)

# Loop over volunteer, reconstruction type (custom vs DICOM)
for volunteer = ["52763", "52782"], prior_type = ["T1"], recon_type = ["custom"]

    # Loading data
    experiment_subname = string(volunteer, "_motion", string(1), "_prior", prior_type, "_", recon_type)
    figures_subfolder = string(figures_folder, experiment_subname, "/"); ~isdir(figures_subfolder) && mkdir(figures_subfolder)
    data_file = string("data_", experiment_subname, ".jld")
    X = load(string(data_folder, data_file))["X"]
    kx, ky, kz = k_coord(X; mesh=true)
    K = cat(reshape(kx, 1, :, 1), reshape(ky, 1, :, 1), reshape(kz, 1, :, 1); dims=3)
    prior = load(string(data_folder, data_file))["prior"]
    ground_truth = load(string(data_folder, data_file))["ground_truth"]

    # Setting Fourier operator
    F = nfft_linop(X, K)
    nt, nk = size(K)
    data = F*ground_truth

    # Multi-scale inversion schedule
    n_scales = 3
    niter_imrecon = ones(Integer, n_scales)
    niter_parest  = ones(Integer, n_scales)
    niter_outloop = 100*ones(Integer, n_scales); niter_outloop[end] = 10;
    ε_schedule = [0.1f0]
    nt, _ = size(K)

    # Setting starting values
    u = deepcopy(ground_truth)
    θ = zeros(Float32, nt, 6)

    # Loop over scales
    damping_factor = nothing
    for (i, scale) in enumerate(n_scales-1:-1:0)

        # Down-scaling the problem (spatially)...
        n_h = div.(X.nsamples, 2^scale)
        X_h = resample(X, n_h)
        K_h = subsample(K, X_h; radial=false)
        F_h = nfft_linop(X_h, K_h)
        nt_h, _ = size(K_h)
        data_h = subsample(K, data, K_h; norm_constant=F.norm_constant/F_h.norm_constant, damping_factor=damping_factor)
        prior_h = resample(prior, n_h; damping_factor=damping_factor)
        u = resample(u, n_h)

        ### Optimization options: ###

        ## Parameter estimation
        opt_parest = parameter_estimation_options(; niter=niter_parest[i], steplength=1f0, reg_matrix=nothing, interp_matrix=nothing)

        ## Image reconstruction
        h = spacing(X_h)
        η = 1f-2*structural_maximum(prior_h; h=h)
        P = structural_weight(prior_h; h=h, η=η, γ=0.92f0)
        opt_inner = FISTA_optimizer(4f0*sum(1 ./h.^2); Nesterov=true, niter=10)
        g = gradient_norm(2, 1, n_h, h, opt_inner; weight=P, complex=true)
        opt_imrecon(ε) = image_reconstruction_options(; prox=indicator(g ≤ ε), Lipschitz_constant=1f0, Nesterov=true, niter=niter_imrecon[i])

        ## Global
        opt(ε) = motion_correction_options(; image_reconstruction_options=opt_imrecon(ε), parameter_estimation_options=opt_parest, niter=niter_outloop[i], niter_estimate_Lipschitz=3)

        ### End optimization options ###

        # Loop over smoothing factor
        for (j, ε_rel) in enumerate(ε_schedule)

            # Joint reconstruction
            @info string("@@@ Scale = ", scale, ", regularization = ", ε_rel)
            θ = reshape(Ip_c2fh*vec(θ), :, 6)
            corrupted_h = F_h(θ)'*data_h
            ε = ε_rel*g(corrupted_h)
            u, θ = motion_corrected_reconstruction(F_h, data_h, u, θ, opt(ε))

        end

        # Up-scaling reconstruction
        u = resample(u, X.nsamples)

    end

    # Loading registered ground-truth
    # @info "@@@ Final rigid registration w/ ground truth"
    # opt_reg = rigid_registration_options(; T=Float32, niter=niter_registration)
    # n = size(ground_truth)
    # u_reg = rigid_registration(u, ground_truth, nothing, opt_reg; spatial_geometry=X)
    # h = spacing(X)
    # opt = FISTA_optimizer(4f0*sum(1 ./h.^2); Nesterov=true, niter=20)
    # g = gradient_norm(2, 1, n, h, opt; complex=true)
    # ε = g(u)
    # u_reg = project(u_reg, ε, g); corrupted_reg = project(corrupted, ε, g); ground_truth_reg = project(ground_truth, ε, g)
    gt_registered_file = string("registered_scans/ground_truth_registered_", volunteer, "_prior", prior_type, "_", recon_type, ".jld")
    ground_truth_reg = load(string(data_folder, gt_registered_file))["ground_truth_reg"]

    # Reconstruction quality
    psnr_recon = psnr(u, ground_truth_reg; preproc=x->abs.(x))
    psnr_conv = psnr(corrupted, ground_truth_reg; preproc=x->abs.(x))
    @info string("@@@ Conventional reconstruction: psnr = ", psnr_conv)
    @info string("@@@ Joint reconstruction: psnr = ", psnr_recon)

    # Save and plot results
    @save string(results_folder, "results_", experiment_subname, ".jld") u corrupted θ psnr_recon psnr_conv
    x, y, z = div.(size(ground_truth), 2).+1
    plot_slices = (VolumeSlice(1, x), VolumeSlice(2, y), VolumeSlice(3, z))
    plot_volume_slices(abs.(u); slices=plot_slices, spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_subfolder, "joint_sTV.png"), orientation=orientation)
    plot_volume_slices(abs.(ground_truth_reg); slices=plot_slices, spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_subfolder, "ground_truth.png"), orientation=orientation)
    plot_volume_slices(abs.(corrupted); slices=plot_slices, spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_subfolder, "corrupted.png"), orientation=orientation)
    plot_parameters(1:size(θ,1), θ, nothing; xlabel="t = phase encoding", vmin=[-10, -10, -10, -10, -10, -10], vmax=[10, 10, 10, 10, 10, 10], fmt1="b", linewidth1=2, savefile=string(figures_subfolder, "motion_pars.png"))

end