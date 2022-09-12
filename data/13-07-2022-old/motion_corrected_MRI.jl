using LinearAlgebra, RetrospectiveMotionCorrectionMRI, ConvexOptimizationUtils, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Experiment name/type
experiment_name = "13-07-2022"; @info string(experiment_name, "\n")

# Setting folder/savefiles
data_folder    = string(pwd(), "/data/", experiment_name, "/")
results_folder = string(pwd(), "/data/", experiment_name, "/results/")
figures_folder = string(pwd(), "/data/", experiment_name, "/figures/")
~isdir(results_folder) && mkdir(results_folder)
~isdir(figures_folder) && mkdir(figures_folder)

# Loading data
X = load(string(data_folder, "data.jld"))["X"]
K = load(string(data_folder, "data.jld"))["K"]
data = load(string(data_folder, "data.jld"))["data"]
struct_prior = true
# struct_prior = false
struct_prior && (prior = load(string(data_folder, "prior.jld"))["prior"])
ground_truth = load(string(data_folder, "ground_truth.jld"))["ground_truth"]
vmin = 0f0; vmax = maximum(abs.(ground_truth))
x, y, z = div.(size(ground_truth), 2).+1
plot_slices = (VolumeSlice(1, x), VolumeSlice(2, y), VolumeSlice(3, z))

# Setting Fourier operator
F = nfft_linop(X, K)
nt, nk = size(K)
u_conventional = F'*data

# Multi-scale inversion schedule
n_scales = 3
niter_imrecon = [1, 1, 1, 1]
niter_parest  = [1, 1, 1, 1]
niter_outloop = [100, 100, 100, 10]
ε_schedule = range(0.1f0, 0.5f0; length=3)
# ε_schedule = range(0.1f0, 0.1f0; length=1)

# Setting starting values
u = deepcopy(u_conventional)
θ = zeros(Float32, nt, 6)

# Loop
for (i, scale) in enumerate(n_scales:-1:0), (j, ε_rel) in enumerate(ε_schedule)
    global u, θ

    # Down-scaling the problem...
    n_h = div.(X.nsamples, 2 .^scale)
    X_h = resample(X, n_h)
    K_h = subsample(K, X_h)
    F_h = nfft_linop(X_h, K_h)
    nt_h, _ = size(K_h)
    data_h = subsample(K, data, K_h; norm_constant=F.norm_constant/F_h.norm_constant)
    u_conventional_h = F_h'*data_h
    struct_prior && (prior_h = resample(prior, n_h))
    u = resample(u, n_h)

    # Optimization options:

        ## Parameter estimation
        ti_h_loc = Float32.(range(1, nt_h; length=size(u_conventional_h,1)))
        # ti_h_loc = Float32.(1:nt_h)
        t_h_loc  = Float32.(1:nt_h)
        Ip_c2f_loc = interpolation1d_motionpars_linop(ti_h_loc, t_h_loc)
        Ip_f2c_loc = interpolation1d_motionpars_linop(t_h_loc, ti_h_loc)
        opt_parest = parameter_estimation_options(; niter=niter_parest[i], steplength=1f0, λ=0f0, cdiag=1f-13, cid=1f10, reg_matrix=nothing, interp_matrix=Ip_c2f_loc, verbose=true, fun_history=true)

        ## Image reconstruction
        h = spacing(X_h)
        struct_prior ? (P = structural_weight(prior_h; h=h, η=1f-2, γ=0.92f0)) : (P = nothing)
        opt_inner = FISTA_optimizer(4f0*sum(1 ./h.^2); Nesterov=true, niter=10)
        g = gradient_norm(2, 1, size(u_conventional_h), h, opt_inner; weight=P, complex=true)
        ε = ε_rel*g(u_conventional_h)
        opt_imrecon = image_reconstruction_options(; prox=indicator(g ≤ ε), Lipschitz_constant=1f0, Nesterov=true, niter=niter_imrecon[i], verbose=true, fun_history=true)

        ## Global
        opt = motion_correction_options(; image_reconstruction_options=opt_imrecon, parameter_estimation_options=opt_parest, niter=niter_outloop[i], niter_estimate_Lipschitz=3, verbose=true, fun_history=true)

    # Selecting motion parameters on low-dimensional space
    idx_θ_h, _ = subsampling_index(K, K_h)
    θ_h = Ip_f2c_loc*vec(θ[idx_θ_h, :])

    # Joint reconstruction
    @info string("\n@@@ Scale = ", scale, ", regularization = ", ε_rel, "\n")
    u, θ_h = motion_corrected_reconstruction(F_h, data_h, u, θ_h, opt)

    # Up-scaling results
    θ[idx_θ_h, :] .= reshape(Ip_c2f_loc*θ_h, nt_h, 6)
    θ .= fill_gaps(idx_θ_h, θ[idx_θ_h, :], nt; average=false, extrapolate=true)
    u = resample(u, X.nsamples)

    # Plot
    plot_volume_slices(abs.(u); slices=plot_slices, spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "temp_e", string(ε_rel), ".png"))
    close("all")

end

# Final rigid registration wrt ground-truth
opt_reg = rigid_registration_options(; T=Float32, niter=20, verbose=true)
u_reg = rigid_registration(u, ground_truth, nothing, opt_reg; spatial_geometry=X)

# Reconstruction quality
unprocessed_file = "unprocessed_scans.jld"
u_conventional = load(string(data_folder, unprocessed_file))["FLAIR_motion"]
ssim_recon = ssim(u_reg/norm(u_reg, Inf), ground_truth/norm(ground_truth, Inf); preproc=x->abs.(x))
psnr_recon = psnr(u_reg/norm(u_reg, Inf), ground_truth/norm(ground_truth, Inf); preproc=x->abs.(x))
ssim_conv = ssim(u_conventional/norm(u_conventional, Inf), ground_truth/norm(ground_truth, Inf); preproc=x->abs.(x))
psnr_conv = psnr(u_conventional/norm(u_conventional, Inf), ground_truth/norm(ground_truth, Inf); preproc=x->abs.(x))
@info string("Conventional reconstruction: ssim = ", ssim_conv, ", psnr = ", psnr_conv)
@info string("Joint reconstruction: ssim = ", ssim_recon, ", psnr = ", psnr_recon)

# Save and plot results
struct_prior ? (extra = "joint_sTV") : (extra = "joint_TV")
@save string(results_folder, "results_", extra, ".jld") u u_reg u_conventional θ ssim_recon psnr_recon ssim_conv psnr_conv
struct_prior && plot_volume_slices(abs.(prior); slices=plot_slices, spatial_geometry=X, vmin=0, vmax=norm(prior, Inf), savefile=string(figures_folder, "prior.png"))
plot_volume_slices(abs.(ground_truth); slices=plot_slices, spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "ground_truth.png"))
plot_volume_slices(abs.(u_conventional); slices=plot_slices, spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "conventional.png"))
plot_volume_slices(abs.(u_reg); slices=plot_slices, spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, extra, ".png"))
plot_parameters(1:size(θ,1), θ, nothing; xlabel="t = phase encoding", vmin=[-10, -10, -10, -10, -10, -10], vmax=[10, 10, 10, 10, 10, 10], fmt1="b", fmt2="r--", linewidth1=2, linewidth2=1, filepath=string(figures_folder, "motion_pars_", extra), ext=".png")