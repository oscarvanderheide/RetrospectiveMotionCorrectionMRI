using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Folders
experiment_name = "26-08-2022-volunteer-study"; @info string(experiment_name, "\n")
exp_folder = string(pwd(), "/data/", experiment_name, "/")
data_folder = string(exp_folder, "data/")
~isdir(data_folder) && mkdir(data_folder)
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)
results_folder = string(exp_folder, "results/")
~isdir(results_folder) && mkdir(results_folder)

# Loop over volunteer, reconstruction type (custom vs DICOM), and motion type
for experiment_subname = ["vol1_priorT1"], motion_type = [3]

    # Loading data
    @info string("Exp: ", experiment_subname, ", motion type: ", motion_type)
    figures_subfolder = string(figures_folder, experiment_subname, "/"); ~isdir(figures_subfolder) && mkdir(figures_subfolder)
    data_file = string("data_", experiment_subname, ".jld")
    X = load(string(data_folder, data_file))["X"]
    K = load(string(data_folder, data_file))["K"]
    data = load(string(data_folder, data_file))[string("data_motion", motion_type)]
    prior = load(string(data_folder, data_file))["prior"]
    ground_truth = load(string(data_folder, data_file))["ground_truth"]
    corrupted = load(string(data_folder, data_file))[string("corrupted_motion", motion_type)]
    vmin = 0f0; vmax = maximum(abs.(ground_truth))
    orientation = load(string(data_folder, data_file))["orientation"]
    mask = load(string(data_folder, data_file))["mask"]

    # Setting Fourier operator
    F = nfft_linop(X, K)
    nt, nk = size(K)

    # Multi-scale inversion schedule
    n_scales = 4
    # n_scales = 1
    niter_imrecon = ones(Integer, n_scales)
    niter_parest  = ones(Integer, n_scales)
    niter_outloop = 100*ones(Integer, n_scales); niter_outloop[end] = 10;
    # niter_outloop = 2*ones(Integer, n_scales); niter_outloop[end] = 2;
    ε_schedule = [0.01f0, 0.1f0, 0.8f0]
    niter_registration = 10
    # niter_registration = 2
    nt, _ = size(K)

    # Setting starting values
    u = deepcopy(corrupted)
    θ = zeros(Float32, nt, 6)

    # Loop over scales
    damping_factor = nothing
    @time for (i, scale) in enumerate(n_scales-1:-1:0)

        # Down-scaling the problem (spatially)...
        n_h = div.(X.nsamples, 2^scale)
        X_h = resample(X, n_h)
        K_h = subsample(K, X_h; radial=false)
        F_h = nfft_linop(X_h, K_h)
        nt_h, _ = size(K_h)
        data_h = subsample(K, data, K_h; norm_constant=F.norm_constant/F_h.norm_constant, damping_factor=damping_factor)
        prior_h = resample(prior, n_h; damping_factor=damping_factor)
        ground_truth_h = resample(ground_truth, n_h; damping_factor=damping_factor)
        mask_h = resample(mask, n_h)
        u = resample(u, n_h)

        # Down-scaling the problem (temporally)...
        nt_h = 50
        t_coarse = Float32.(range(1, nt; length=nt_h))
        t_fine = Float32.(1:nt)
        # interp = :spline
        interp = :linear
        Ip_c2f = interpolation1d_motionpars_linop(t_coarse, t_fine; interp=interp)
        Ip_f2c = interpolation1d_motionpars_linop(t_fine, t_coarse; interp=interp)
        t_fine_h = K_h.subindex_phase_encoding

        ### Optimization options: ###

        ## Parameter estimation
        scaling_diagonal = 1f-3
        scaling_mean     = 1f-4
        scaling_id       = 0f0
        Ip_c2fh = interpolation1d_motionpars_linop(t_coarse, Float32.(t_fine_h); interp=interp)
        D = derivative1d_motionpars_linop(t_coarse, 1; pars=(true, true, true, true, true, true))
        λ = 1f-3*sqrt(norm(data_h)^2/spectral_radius(D'*D; niter=3))
        opt_parest = parameter_estimation_options(; niter=niter_parest[i], steplength=1f0, λ=λ, scaling_diagonal=scaling_diagonal, scaling_mean=scaling_mean, scaling_id=scaling_id, reg_matrix=D, interp_matrix=Ip_c2fh)

        ## Image reconstruction
        h = spacing(X_h)
        η = 1f-2*structural_maximum(prior_h; h=h)
        P = structural_weight(prior_h; h=h, η=η, γ=1f0)
        # P = nothing
        opt_inner = FISTA_options(4f0*sum(1f0./h.^2); Nesterov=true, niter=10)
        C0 = zero_set(ComplexF32, (!).(mask_h))
        g = gradient_norm(2, 1, n_h, h; weight=P, complex=true, options=opt_inner)
        # g1 = set_options(g+indicator(C0), opt_inner)
        g1 = g #########################
        opt_imrecon(ε) = image_reconstruction_options(; prox=indicator(g1 ≤ ε), Nesterov=true, niter=niter_imrecon[i])

        ## Global
        opt(ε) = motion_correction_options(; image_reconstruction_options=opt_imrecon(ε), parameter_estimation_options=opt_parest, niter=niter_outloop[i], niter_estimate_Lipschitz=3, verbose=false)

        ### End optimization options ###

        # Loop over smoothing factor
        for (j, ε_rel) in enumerate(ε_schedule)

            # Selecting motion parameters on low-dimensional space
            θ_coarse = reshape(Ip_f2c*vec(θ), :, 6)
            # θ_coarse = reshape(Float32.(Ip_c2f\vec(θ)), :, 6)

            # Joint reconstruction
            @info string("@@@ Scale = ", scale, ", regularization = ", ε_rel)
            θ_h = reshape(Ip_c2fh*vec(θ_coarse), :, 6)
            corrupted_h = F_h(θ_h)'*data_h
            ε = ε_rel*g(corrupted_h)
            # ε = ε_rel*g(ground_truth_h) # inverse crime for conventional TV!
            u, θ_coarse = motion_corrected_reconstruction(F_h, data_h, u, θ_coarse, opt(ε))

            # Up-scaling motion parameters
            θ .= reshape(Ip_c2f*vec(θ_coarse), :, 6)
            n_avg = div(size(θ, 1), 10)
            θ[1:t_fine_h[1]+n_avg-1,:] .= sum(θ[t_fine_h[1]:t_fine_h[1]+n_avg-1,:]; dims=1)/n_avg
            θ[t_fine_h[end]-n_avg+1:end,:] .= sum(θ[t_fine_h[end]-n_avg+1:t_fine_h[end],:]; dims=1)/n_avg

            # Plot
            plot_volume_slices(abs.(u); spatial_geometry=X_h, vmin=vmin, vmax=vmax, savefile=string(figures_subfolder, "temp.png"), orientation=orientation)
            close("all")
            plot_parameters(1:nt, θ, nothing; xlabel="t = phase encoding", vmin=[-10, -10, -10, -10, -10, -10], vmax=[10, 10, 10, 10, 10, 10], fmt1="b", linewidth1=2, savefile=string(figures_subfolder, "temp_motion_pars.png"))
            close("all")

        end

        # Up-scaling reconstruction
        u = resample(u, X.nsamples)

    end

    # Denoising ground-truth/corrupted volumes
    @info "@@@ Post-processing figures"
    opt_reg = rigid_registration_options(; niter=niter_registration, verbose=false)
    u_reg, _ = rigid_registration(u, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=5)
    corrupted_reg, _ = rigid_registration(corrupted, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=5)
    opt_postproc = FISTA_options(4f0*sum(1f0./spacing(X).^2); Nesterov=true, niter=10)
    g = gradient_norm(2, 1, size(ground_truth), spacing(X); complex=true)
    C = zero_set(ComplexF32, (!).(mask))
    ε_reg = 0.8f0*g(ground_truth)
    u_reg = proj(u_reg, g(u), g, opt_postproc); u_reg = proj(u_reg, C)
    corrupted_reg    = proj(corrupted_reg, ε_reg, g, opt_postproc); corrupted_reg = proj(corrupted_reg, C)
    ground_truth_reg = proj(ground_truth,  ε_reg, g, opt_postproc); ground_truth_reg = proj(ground_truth_reg, C)

    # Reconstruction quality
    nx, ny, nz = size(ground_truth)[[invperm(orientation.perm)...]]
    slices = (VolumeSlice(1, div(nx,2)+1, nothing),
              VolumeSlice(2, div(ny,2)+1, nothing),
              VolumeSlice(3, div(nz,2)+1, nothing))
    fact = norm(ground_truth, Inf)
    psnr_recon = psnr(abs.(u_reg)/fact, abs.(ground_truth_reg)/fact; slices=slices, orientation=orientation)
    psnr_conv = psnr(abs.(corrupted_reg)/fact, abs.(ground_truth_reg)/fact; slices=slices, orientation=orientation)
    ssim_recon = ssim(abs.(u_reg)/fact, abs.(ground_truth_reg)/fact; slices=slices, orientation=orientation)
    ssim_conv = ssim(abs.(corrupted_reg)/fact, abs.(ground_truth_reg)/fact; slices=slices, orientation=orientation)
    @info string("@@@ Conventional reconstruction: psnr = ", psnr_conv, ", ssim = ", ssim_conv)
    @info string("@@@ Joint reconstruction: psnr = ", psnr_recon, ", ssim = ", ssim_recon)

    # Save and plot results
    @save string(results_folder, "results_", experiment_subname, "_motion", motion_type, ".jld") u θ psnr_recon psnr_conv ssim_recon ssim_conv u_reg corrupted_reg ground_truth_reg
    plot_volume_slices(abs.(u_reg); spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_subfolder, "corrected_motion", string(motion_type), ".png"), orientation=orientation)
    plot_volume_slices(abs.(ground_truth_reg); spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_subfolder, "ground_truth_reg.png"), orientation=orientation)
    plot_volume_slices(abs.(corrupted_reg); spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_subfolder, "corrupted_reg_motion", string(motion_type), ".png"), orientation=orientation)
    θ_min =  minimum(θ; dims=1); θ_max = maximum(θ; dims=1); Δθ = θ_max-θ_min; θ_middle = (θ_min+θ_max)/2
    Δθ = [ones(1,3)*max(Δθ[1:3]...)/2 ones(1,3)*max(Δθ[4:end]...)/2]
    plot_parameters(1:size(θ,1), θ, nothing; xlabel="t = phase encoding", vmin=vec(θ_middle-1.1f0*Δθ), vmax=vec(θ_middle+1.1f0*Δθ), fmt1="b", linewidth1=2, savefile=string(figures_subfolder, "parameters_motion", string(motion_type), ".png"))
    close("all")

end