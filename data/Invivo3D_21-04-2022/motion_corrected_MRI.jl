using BlindMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, AbstractLinearOperators, PyPlot, JLD, ImageFiltering, LinearAlgebra
include(string(pwd(), "/scripts/plot_results.jl"))

# Experiment name/type
experiment_name = "Invivo3D_21-04-2022"; println(experiment_name, "\n")

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

# Setting Fourier operator
F = nfft(X, K; tol=1f-6)
nt, nk = size(K)
u_conventional = F(zeros(Float32,nt,6))'*data

# Multi-scale inversion
u = u_conventional
θ = zeros(Float32, nt, 6)
n_scales = 2
niter_imrecon = 1*ones(Int64, n_scales+1)
niter_parest  = [1, 1, 1]
niter_outloop = [100, 100, 100]
# niter_imrecon = 10*ones(Int64, n_scales+1)
# niter_parest  = [1, 1, 1]
# niter_outloop = [10, 10, 10]
ε_schedule = [0.1f0, 0.5f0, 0.8f0]
# ε_schedule = [0.1f0]
for (i, scale) in enumerate(n_scales:-1:0), (j, ε_rel) in enumerate(ε_schedule)
    global u, θ

    # Down-scaling the problem...
    F_h              = downscale(F;                 fact=scale); nt_h, _ = size(F_h.K)
    data_h           = downscale(data,           K; fact=scale, flat=true)
    u_conventional_h = F_h(zeros(Float32, nt_h, 6))'*data_h
    struct_prior && (prior_h = downscale(prior, X; fact=scale, flat=true))
    u = downscale(u, X; fact=scale, flat=true)

    # Optimization options:

        ## Parameter estimation
        ti_h_loc = Float32.(range(1, nt_h; length=size(u_conventional_h,1)))
        # ti_h_loc = Float32.(1:nt_h)
        t_h_loc  = Float32.(1:nt_h)
        Ip_c2f_loc = interpolation1d_motionpars_linop(ti_h_loc, t_h_loc)
        Ip_f2c_loc = interpolation1d_motionpars_linop(t_h_loc, ti_h_loc)
        loss = data_residual_loss(ComplexF32, 2, 2)
        opt_parest = parameter_estimation_options(Float32; loss=loss, niter=niter_parest[i], steplength=1f0, λ=0f0, cdiag=1f-13, cid=1f10, reg_matrix=nothing, interp_matrix=Ip_c2f_loc, verbose=true)

        ## Image reconstruction
        struct_prior ? (P = structural_weight(prior_h; η=1f-2, γ=0.92f0)) : (P = nothing)
        g = gradient_norm(2, 1, size(u_conventional_h), (1f0,1f0,1f0); weight=P, T=ComplexF32)
        ε = ε_rel*g(u_conventional_h)
        opt_proj = opt_fista(1f0/12f0; niter=5, Nesterov=true)
        prox(u, _) = project(u, ε, g, opt_proj)
        opt_recon = image_reconstruction_FISTA_options(Float32; loss=loss, niter=niter_imrecon[i], steplength=nothing, niter_EstLipschitzConst=3, prox=prox, Nesterov=true, verbose=true)

        ## Global
        opt = motion_correction_options(; niter=niter_outloop[i], image_reconstruction_options=opt_recon, parameter_estimation_options=opt_parest, verbose=true)

    # Selecting motion parameters on low-dimensional space
    idx_θ_h = downscale_phase_encode_index(K; fact=scale)
    θ_h = Ip_f2c_loc*vec(θ[idx_θ_h, :])

    # Joint reconstruction
    println("\nScale = ", scale, ", regularization = ", ε_rel, "\n")
    u, θ_h, fval = motion_corrected_reconstruction(F_h, data_h, u, θ_h, opt)

    # Up-scaling results
    θ[idx_θ_h, :] .= reshape(Ip_c2f_loc*θ_h, nt_h, 6)
    θ = interp_linear_filling(X.n[[K.phase_encoding...]], θ, n_scales-i+1; keep_low_freqs=true, extrapolate=true)
    u = upscale(u, F_h.X; fact=n_scales-i+1)

    # Plot
    x, y, z = div.(size(u), 2).+1
    plot_3D_result(u, 0, maximum(abs.(u)); x=x, y=y, z=z, filepath=string(figures_folder, "temp_e", string(ε_rel)), ext=".png")
    close("all")

end

# Final rigid registration wrt ground-truth
ground_truth = load(string(data_folder, "ground_truth.jld"))["ground_truth"]
vmin = 0; vmax = maximum(abs.(ground_truth))
opt_reg = rigid_registration_options(Float32; niter=20, verbose=true)
u, _, _ = rigid_registration(u, ground_truth, nothing, opt_reg)

# # Reconstruction quality
# x, y, z = div.(size(u), 2).+1
# ssim_recon = ssim(u, ground_truth; x=x, y=y, z=z)
# psnr_recon = psnr(u, ground_truth; x=x, y=y, z=z)
# println("Joint reconstruction: ssim_recon = ", ssim_recon, ", psnr_recon = ", psnr_recon)

# Save and plot results
# x, y, z = div.(size(ground_truth),2).+1
# struct_prior ? (extra = "joint_sTV") : (extra = "joint_TV")
# @save string(results_folder, "results_", extra, ".jld") u θ# ssim_recon psnr_recon
# struct_prior && plot_3D_result(prior, minimum(abs.(prior)), maximum(abs.(prior)); x=x, y=y, z=z, filepath=string(figures_folder, "prior"), ext=".png")
# plot_3D_result(ground_truth, 0, maximum(abs.(ground_truth)); x=x, y=y, z=z, filepath=string(figures_folder, "ground_truth"), ext=".png")
# plot_3D_result(u_conventional, 0, maximum(abs.(u_conventional)); x=x, y=y, z=z, filepath=string(figures_folder, "conventional"), ext=".png")
# plot_3D_result(u, 0, maximum(abs.(u)); x=x, y=y, z=z, filepath=string(figures_folder, extra), ext=".png")
# plot_parameters(1:size(θ,1), θ, nothing; xlabel="t = phase encoding", vmin=[-10, -10, -10, -1, -1, -1], vmax=[3, 3, 3, 10, 10, 10], fmt1="b", fmt2="r--", linewidth1=2, linewidth2=1, filepath=string(figures_folder, "motion_pars_", extra), ext=".png")
