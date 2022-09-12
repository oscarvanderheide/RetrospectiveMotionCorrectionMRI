using LinearAlgebra, RetrospectiveMotionCorrectionMRI, ConvexOptimizationUtils, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Folders & files
experiment_name = "18-08-2022-patient"; @info string(experiment_name, "\n")
exp_folder = string(pwd(), "/data/", experiment_name, "/")
unprocessed_scans_folder = string(exp_folder, "unprocessed_scans/")
data_folder = string(exp_folder, "data/")
~isdir(data_folder) && mkdir(data_folder)
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)
results_folder = string(exp_folder, "results/")
~isdir(results_folder) && mkdir(results_folder)
data_file = "data.jld"
X = load(string(data_folder, data_file))["X"]
K = load(string(data_folder, data_file))["K"]
data = load(string(data_folder, data_file))["data"]
prior = load(string(data_folder, data_file))["prior"]
ground_truth = load(string(data_folder, data_file))["ground_truth"]
corrupted = load(string(data_folder, data_file))["corrupted"]
vmin = 0f0; vmax = maximum(abs.(ground_truth))
orientation = load(string(data_folder, data_file))["orientation"]

# Setting Fourier operator
F = nfft_linop(X, K)
nt, _ = size(K)

# Multi-scale inversion schedule
n_scales = 3
# n_scales = 4
niter_imrecon = ones(Integer, n_scales)
niter_parest  = ones(Integer, n_scales)
niter_outloop = 100*ones(Integer, n_scales); niter_outloop[end] = 10;
ε_schedule = range(0.1f0, 0.5f0; length=3)
niter_registration = 20

# Setting starting values
# u = zeros(ComplexF32, size(X))
u = deepcopy(corrupted)
θ = zeros(Float32, nt, 6)

# Loop over scales
damping_factor = nothing
for (i, scale) in enumerate(n_scales-1:-1:0)
    global u, θ

    # Down-scaling the problem...
    n_h = div.(X.nsamples, 2^scale)
    # n_h = [div.(X.nsamples, 2^scale)...]; n_h[readout_dim(K)] = X.nsamples[readout_dim(K)]; n_h = tuple(n_h...) ###
    X_h = resample(X, n_h)
    K_h = subsample(K, X_h; radial=true)
    # K_h = subsample(K, X_h; radial=false, also_readout=false) ###
    F_h = nfft_linop(X_h, K_h)
    nt_h, _ = size(K_h)
    data_h = subsample(K, data, K_h; norm_constant=F.norm_constant/F_h.norm_constant, damping_factor=damping_factor)
    corrupted_h = F_h'*data_h
    prior_h = resample(prior, n_h; damping_factor=damping_factor)
    u = resample(u, n_h)
    idx_θ_h = K_h.subindex_phase_encoding

    ### Optimization options: ###

    ## Parameter estimation
    t_global = Float32.(1:nt)
    t_local = t_global[idx_θ_h]
    D = derivative1d_motionpars_linop(t_local, 1)
    ρ = spectral_radius(D'*D; niter=3)
    λ = 1f0*norm(data_h)/sqrt(ρ)
    # opt_parest = parameter_estimation_options(; niter=niter_parest[i], steplength=1f0, λ=λ, cdiag=1f-13, cid=1f10, reg_matrix=D, interp_matrix=nothing)
    opt_parest = parameter_estimation_options(; niter=niter_parest[i], steplength=1f0, λ=λ, cdiag=1f-3, cid=1f-3*norm(data_h)^2 ./(1f0, 1f0, 1f0, Float32(pi/180), Float32(pi/180), Float32(pi/180)).^2, reg_matrix=D, interp_matrix=nothing)

    ## Image reconstruction
    h = spacing(X_h)
    η = 1f-2
    P = structural_weight(prior_h; h=h, η=η, γ=0.92f0)
    opt_inner = FISTA_optimizer(4f0*sum(1 ./h.^2); Nesterov=true, niter=10)
    g = gradient_norm(2, 1, size(corrupted_h), h, opt_inner; weight=P, complex=true)
    opt_imrecon(ε) = image_reconstruction_options(; prox=indicator(g ≤ ε), Lipschitz_constant=1f0, Nesterov=true, niter=niter_imrecon[i])

    ## Global
    opt(ε) = motion_correction_options(; image_reconstruction_options=opt_imrecon(ε), parameter_estimation_options=opt_parest, niter=niter_outloop[i], niter_estimate_Lipschitz=3)

    ### End optimization options ###

    # Loop over smoothing factor
    for (j, ε_rel) in enumerate(ε_schedule)

        # Selecting motion parameters on low-dimensional space
        θ_h = θ[idx_θ_h, :]

        # Joint reconstruction
        @info string("@@@ Scale = ", scale, ", regularization = ", ε_rel)
        ε = ε_rel*g(corrupted_h)
        u, θ_h = motion_corrected_reconstruction(F_h, data_h, u, θ_h, opt(ε))
        θ[idx_θ_h, :] .= reshape(θ_h, :, 6)
        # θ .= fill_gaps(idx_θ_h, θ[idx_θ_h, :], size(θ,1); average=false, extrapolate=true, smoothing=50f0)
        θ .= fill_gaps(idx_θ_h, θ[idx_θ_h, :], size(θ,1); average=false, extrapolate=true)

        # Plot
        plot_volume_slices(abs.(u); spatial_geometry=X_h, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "temp.png"), orientation=orientation)
        close("all")
        plot_parameters(1:size(θ,1), θ, nothing; xlabel="t = phase encoding", vmin=[-10, -10, -10, -10, -10, -10], vmax=[10, 10, 10, 10, 10, 10], fmt1="b", linewidth1=2, savefile=string(figures_folder, "temp_motion_pars.png"))

    end

    # Up-scaling reconstruction
    u = resample(u, X.nsamples)

end

# Final rigid registration wrt ground-truth
@info "@@@ Final rigid registration w/ ground truth"
opt_reg = rigid_registration_options(; T=Float32, niter=niter_registration)
u_reg = rigid_registration(u, ground_truth, nothing, opt_reg; spatial_geometry=X)

# Reconstruction quality
ssim_recon = ssim(u_reg/norm(u_reg, Inf), ground_truth/norm(ground_truth, Inf); preproc=x->abs.(x))
psnr_recon = psnr(u_reg/norm(u_reg, Inf), ground_truth/norm(ground_truth, Inf); preproc=x->abs.(x))
ssim_conv = ssim(corrupted/norm(corrupted, Inf), ground_truth/norm(ground_truth, Inf); preproc=x->abs.(x))
psnr_conv = psnr(corrupted/norm(corrupted, Inf), ground_truth/norm(ground_truth, Inf); preproc=x->abs.(x))
@info string("@@@ Conventional reconstruction: ssim = ", ssim_conv, ", psnr = ", psnr_conv)
@info string("@@@ Joint reconstruction: ssim = ", ssim_recon, ", psnr = ", psnr_recon)

# Save and plot results
@save string(results_folder, "results.jld") u u_reg corrupted θ ssim_recon psnr_recon ssim_conv psnr_conv
plot_volume_slices(abs.(u_reg); spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "joint_sTV.png"), orientation=orientation)
plot_parameters(1:size(θ,1), θ, nothing; xlabel="t = phase encoding", vmin=[-10, -10, -10, -10, -10, -10], vmax=[10, 10, 10, 10, 10, 10], fmt1="b", linewidth1=2, savefile=string(figures_folder, "motion_pars.png"))