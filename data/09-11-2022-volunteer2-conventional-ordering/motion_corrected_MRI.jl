using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Folders
experiment_name = "09-11-2022-volunteer2-conventional-ordering"; @info string(experiment_name, "\n")
exp_folder = string(pwd(), "/data/", experiment_name, "/")
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)
results_folder = string(exp_folder, "results/")
~isdir(results_folder) && mkdir(results_folder)

# Loop over motion type
for motion_type = [2]

    # Loading data
    @info string("Exp: ", experiment_name, ", motion type: ", motion_type)
    data_file = "data.jld"
    X, K, data, prior, ground_truth, corrupted, orientation, mask = load(string(exp_folder, data_file), "X", "K", string("data_motion", motion_type), "prior", "ground_truth", string("corrupted_motion", motion_type), "orientation", "mask_prior")

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
    ε_schedule = [0.01f0, 0.1f0, 0.2f0]
    niter_registration = 20
    # niter_registration = 2
    nt, _ = size(K)
    t_fine = Float32.(1:nt)

    # Setting starting values
    u = deepcopy(corrupted)
    θ = zeros(Float32, nt, 6)

    # Loop over scales
    damping_factor = nothing
    for (i, scale) in enumerate(n_scales-1:-1:0)

        # Down-scaling the problem (spatially)...
        n_h = div.(X.nsamples, 2^scale)
        X_h = resample(X, n_h)
        K_h = subsample(K, X_h; radial=false)
        F_h = nfft_linop(X_h, K_h)
        data_h = subsample(K, data, K_h; norm_constant=F.norm_constant/F_h.norm_constant, damping_factor=damping_factor)
        prior_h = resample(prior, n_h; damping_factor=damping_factor)
        mask_h = resample(mask, n_h)
        u = resample(u, n_h)

        # Down-scaling the problem (temporally)...
        nt_coarse = 25
        # nt_coarse = nt
        t_coarse = Float32.(range(1, nt; length=nt_coarse))
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
        opt_inner = FISTA_options(4f0*sum(1f0./h.^2); Nesterov=true, niter=10)
        C0 = zero_set(ComplexF32, (!).(mask_h))
        g = gradient_norm(2, 1, n_h, h; weight=P, complex=true, options=opt_inner)
        opt_imrecon(ε) = image_reconstruction_options(; prox=indicator(g ≤ ε), Nesterov=true, niter=niter_imrecon[i])

        ## Global
        opt(ε) = motion_correction_options(; image_reconstruction_options=opt_imrecon(ε), parameter_estimation_options=opt_parest, niter=niter_outloop[i], niter_estimate_Lipschitz=3, verbose=false)

        ### End optimization options ###

        # Loop over smoothing factor
        for (j, ε_rel) in enumerate(ε_schedule)

            # Selecting motion parameters on low-dimensional space
            θ_coarse = reshape(Ip_f2c*vec(θ), :, 6)

            # Joint reconstruction
            @info string("@@@ Scale = ", scale, ", regularization = ", ε_rel)
            θ_h = reshape(Ip_c2fh*vec(θ_coarse), :, 6)
            corrupted_h = F_h(θ_h)'*data_h
            ε = ε_rel*g(corrupted_h)
            u, θ_coarse = motion_corrected_reconstruction(F_h, data_h, u, θ_coarse, opt(ε))

            # Up-scaling motion parameters
            θ .= reshape(Ip_c2f*vec(θ_coarse), :, 6)
            n_avg = div(size(θ, 1), 10)
            θ[1:t_fine_h[1]+n_avg-1,:] .= sum(θ[t_fine_h[1]:t_fine_h[1]+n_avg-1,:]; dims=1)/n_avg
            θ[t_fine_h[end]-n_avg+1:end,:] .= sum(θ[t_fine_h[end]-n_avg+1:t_fine_h[end],:]; dims=1)/n_avg

            # Plot
            plot_volume_slices(abs.(u); spatial_geometry=X_h, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(figures_folder, "temp.png"), orientation=orientation)
            θ_min =  minimum(θ; dims=1); θ_max = maximum(θ; dims=1); Δθ = θ_max-θ_min; θ_middle = (θ_min+θ_max)/2
            Δθ = [ones(1,3)*max(Δθ[1:3]...)/2 ones(1,3)*max(Δθ[4:end]...)/2]
            plot_parameters(1:nt, θ, nothing; xlabel="t = phase encoding", vmin=vec(θ_middle-1.1f0*Δθ), vmax=vec(θ_middle+1.1f0*Δθ), fmt1="b", linewidth1=2, savefile=string(figures_folder, "temp_motion_pars.png"))
            close("all")

        end

    end

    # Denoising ground-truth/corrupted volumes
    @info "@@@ Post-processing figures"
    corrupted_reg, mask = load(string(exp_folder, data_file), string("corrupted_motion", motion_type, "_reg"), "mask_ground_truth")
    C = zero_set(ComplexF32, (!).(mask))
    corrupted_reg = proj(corrupted_reg, C)
    opt_reg = rigid_registration_options(; niter=niter_registration, verbose=true)
    u_reg, _ = rigid_registration(u, ground_truth, nothing, opt_reg; spatial_geometry=X, nscales=5); u_reg = proj(u_reg, C)

    # Reconstruction quality
    fact = norm(ground_truth, Inf)
    psnr_corrected = psnr(abs.(u_reg)/fact, abs.(ground_truth)/fact)
    psnr_corrupted = psnr(abs.(corrupted_reg)/fact, abs.(ground_truth)/fact)
    ssim_corrected = ssim(abs.(u_reg)/fact, abs.(ground_truth)/fact)
    ssim_corrupted = ssim(abs.(corrupted_reg)/fact, abs.(ground_truth)/fact)
    @info string("@@@ Conventional reconstruction: psnr = ", psnr_corrupted, ", ssim = ", ssim_corrupted)
    @info string("@@@ Joint reconstruction: psnr = ", psnr_corrected, ", ssim = ", ssim_corrected)

    # Save and plot results
    @save string(results_folder, "results_motion", motion_type, ".jld") u u_reg θ psnr_corrected psnr_corrupted ssim_corrected ssim_corrupted
    plot_volume_slices(abs.(u_reg); spatial_geometry=X, vmin=0f0, vmax=norm(ground_truth,Inf), savefile=string(figures_folder, "corrected_motion", string(motion_type), ".png"), orientation=orientation)
    θ_min =  minimum(θ; dims=1); θ_max = maximum(θ; dims=1); Δθ = θ_max-θ_min; θ_middle = (θ_min+θ_max)/2
    Δθ = [ones(1,3)*max(Δθ[1:3]...)/2 ones(1,3)*max(Δθ[4:end]...)/2]
    plot_parameters(1:size(θ,1), θ, nothing; xlabel="t = phase encoding", vmin=vec(θ_middle-1.1f0*Δθ), vmax=vec(θ_middle+1.1f0*Δθ), fmt1="b", linewidth1=2, savefile=string(figures_folder, "parameters_motion", string(motion_type), ".png"))
    close("all")

end