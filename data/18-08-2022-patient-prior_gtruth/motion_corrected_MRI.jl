using LinearAlgebra, RetrospectiveMotionCorrectionMRI, ConvexOptimizationUtils, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Folders
experiment_name = "18-08-2022-patient-prior_gtruth"; @info string(experiment_name, "\n")
exp_folder = string(pwd(), "/data/", experiment_name, "/")
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)
results_folder = string(exp_folder, "results/")
~isdir(results_folder) && mkdir(results_folder)

# Loading data
data_file = "data.jld"
X = load(string(exp_folder, data_file))["X"]
K = load(string(exp_folder, data_file))["K"]
data = load(string(exp_folder, data_file))["data"]
prior = load(string(exp_folder, data_file))["prior"]
ground_truth = load(string(exp_folder, data_file))["ground_truth"]
corrupted = load(string(exp_folder, data_file))["corrupted"]
vmin = 0f0; vmax = maximum(abs.(ground_truth))
orientation = load(string(exp_folder, data_file))["orientation"]

# Setting Fourier operator
F = nfft_linop(X, K)
nt, nk = size(K)

# Multi-scale inversion schedule
n_scales = 3
niter_imrecon = [1, 1, 1]
niter_parest  = [1, 1, 1]
niter_outloop = [100, 100, 10]
ε_schedule = range(0.1f0, 0.5f0; length=3)
niter_registration = 20

# Setting starting values
u = zeros(ComplexF32, size(X))
θ = zeros(Float32, nt, 6)

# Loop over scales
damping_factor = nothing
for (i, scale) in enumerate(n_scales-1:-1:0)
    global u, θ

    # Down-scaling the problem...
    n_h = div.(X.nsamples, 2 .^scale)
    X_h = resample(X, n_h)
    K_h = subsample(K, X_h; radial=true)
    F_h = nfft_linop(X_h, K_h)
    nt_h, _ = size(K_h)
    data_h = subsample(K, data, K_h; norm_constant=F.norm_constant/F_h.norm_constant, damping_factor=damping_factor)
    corrupted_h = F_h'*data_h
    prior_h = resample(prior, n_h; damping_factor=damping_factor)
    u = resample(u, n_h)

    # Optimization options:

        ## Parameter estimation
        ti_h_loc = Float32.(range(1, nt_h; length=size(corrupted_h,3)))
        # ti_h_loc = Float32.(1:nt_h)
        t_h_loc  = Float32.(1:nt_h)
        Ip_c2f_loc = interpolation1d_motionpars_linop(ti_h_loc, t_h_loc)
        Ip_f2c_loc = interpolation1d_motionpars_linop(t_h_loc, ti_h_loc)
        opt_parest = parameter_estimation_options(; niter=niter_parest[i], steplength=1f0, λ=0f0, cdiag=1f-13, cid=1f10, reg_matrix=nothing, interp_matrix=Ip_c2f_loc)

        ## Image reconstruction
        h = spacing(X_h)
        P = structural_weight(prior_h; h=h, η=1f-2, γ=0.92f0)
        opt_inner = FISTA_optimizer(4f0*sum(1 ./h.^2); Nesterov=true, niter=10)
        g = gradient_norm(2, 1, size(corrupted_h), h, opt_inner; weight=P, complex=true)
        opt_imrecon(ε) = image_reconstruction_options(; prox=indicator(g ≤ ε), Lipschitz_constant=1f0, Nesterov=true, niter=niter_imrecon[i])

        ## Global
        opt(ε) = motion_correction_options(; image_reconstruction_options=opt_imrecon(ε), parameter_estimation_options=opt_parest, niter=niter_outloop[i], niter_estimate_Lipschitz=3)

    # Selecting motion parameters on low-dimensional space
    idx_θ_h = K_h.subindex_phase_encoding
    θ_h = Ip_f2c_loc*vec(θ[idx_θ_h, :])

    # Loop over smoothing factor
    for (j, ε_rel) in enumerate(ε_schedule)

        # Joint reconstruction
        @info string("@@@ Scale = ", scale, ", regularization = ", ε_rel)
        ε = ε_rel*g(corrupted_h)
        u, θ_h = motion_corrected_reconstruction(F_h, data_h, u, θ_h, opt(ε))
        θ[idx_θ_h, :] .= reshape(Ip_c2f_loc*θ_h, nt_h, 6)

        # Plot
        plot_volume_slices(abs.(u); spatial_geometry=X_h, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "temp.png"), orientation=orientation)
        plot_parameters(1:size(θ,1), θ, nothing; xlabel="t = phase encoding", vmin=[-10, -10, -10, -10, -10, -10], vmax=[10, 10, 10, 10, 10, 10], fmt1="b", fmt2="r--", linewidth1=2, linewidth2=1, filepath=string(figures_folder, "temp_motion_pars"), ext=".png")
        close("all")

    end

    # Up-scaling results
    (scale != 0) && (θ .= fill_gaps(idx_θ_h, θ[idx_θ_h, :], nt; average=false, extrapolate=true, smoothing=Float32(size(corrupted_h,3))))
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
@save string(results_folder, "results_joint_sTV.jld") u u_reg corrupted θ ssim_recon psnr_recon ssim_conv psnr_conv
x, y, z = div.(size(ground_truth), 2).+1
plot_slices = (VolumeSlice(1, x), VolumeSlice(2, y), VolumeSlice(3, z))
plot_volume_slices(abs.(u_reg); slices=plot_slices, spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "joint_sTV.png"), orientation=orientation)
plot_parameters(1:size(θ,1), θ, nothing; xlabel="t = phase encoding", vmin=[-10, -10, -10, -10, -10, -10], vmax=[10, 10, 10, 10, 10, 10], fmt1="b", fmt2="r--", linewidth1=2, linewidth2=1, filepath=string(figures_folder, "motion_pars"), ext=".png")