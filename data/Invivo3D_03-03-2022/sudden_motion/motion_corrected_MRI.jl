using BlindMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, AbstractLinearOperators, PyPlot, JLD, ImageFiltering, LinearAlgebra
include(string(pwd(), "/scripts/plot_results.jl"))
include(string(pwd(), "/envelope.jl"))

# Experiment name/type
phantom = "Invivo3D_03-03-2022"; motion = "sudden_motion"
experiment_name = string(phantom, "/", motion)
println(experiment_name, "\n")

# Setting folder/savefiles
phantom_folder = string(pwd(), "/data/", phantom, "/")
data_folder    = string(pwd(), "/data/", experiment_name, "/")
results_folder = string(pwd(), "/data/", experiment_name, "/results/")
figures_folder = string(pwd(), "/data/", experiment_name, "/figures/")
~isdir(results_folder) && mkdir(results_folder)
~isdir(figures_folder) && mkdir(figures_folder)

# Loading data
ground_truth = load(string(phantom_folder, "ground_truth.jld"))["ground_truth"]
X = load(string(data_folder, "data.jld"))["X"]
K = load(string(data_folder, "data.jld"))["K"]
data = load(string(data_folder, "data.jld"))["data"]
ground_truth = load(string(phantom_folder, "ground_truth.jld"))["ground_truth"]
vmin = 0; vmax = maximum(abs.(ground_truth))

# Structure-guided prior
struct_prior = true
# struct_prior = false
struct_prior && (prior = load(string(phantom_folder, "prior.jld"))["prior"])

# Setting Fourier operator
F = nfft(X, K; tol=1f-6)
nt, nk = size(K)
u_conventional = F(zeros(Float32,nt,6))'*data

# Multi-scale inversion
u = u_conventional
θ = zeros(Float32, nt, 6)
n_scales = 4
niter_imrecon = 10*ones(Int64, n_scales+1)
niter_parest  = [1, 1, 1, 1, 1]
niter_outloop = [10, 10, 10, 10, 1]
# ε_schedule = [0.01f0, 0.1f0, 0.5f0]
ε_schedule = [0.01f0]#, 0.1f0, 0.5f0]
for ε_rel in ε_schedule, (i, scale) in enumerate(n_scales:-1:1)
    global u

    # Down-scaling the problem...
    F_h              = downscale(F;                 fact=scale); nt_h, _ = size(F_h.K)
    data_h           = downscale(data,           K; fact=scale, flat=true)
    u_conventional_h = F_h(zeros(Float32, nt_h, 6))'*data_h
    struct_prior && (prior_h = downscale(prior, X; fact=scale, flat=true))
    u = downscale(u, X; fact=scale, flat=true)

    # Optimization options:
    ## Parameter estimation
    # ti_h_loc = Float32.(range(1, nt_h; length=51))
    # ti_h_loc = [Float32.(1:div(nt_h,2)); Float32.(range(div(nt_h,2)+1, nt_h; length=size(u,1)))]
    ti_h_loc = Float32.(1:nt_h)
    t_h_loc  = Float32.(1:nt_h)
    # Ip_c2f_loc = interpolation1d_motionpars_linop((t_h_loc,t_h_loc,t_h_loc,ti_h_loc,t_h_loc,t_h_loc), t_h_loc)
    Ip_c2f_loc = interpolation1d_motionpars_linop(ti_h_loc, t_h_loc)
    # Ip_f2c_loc = interpolation1d_motionpars_linop(t_h_loc, (t_h_loc,t_h_loc,t_h_loc,ti_h_loc,t_h_loc,t_h_loc))
    Ip_f2c_loc = interpolation1d_motionpars_linop(t_h_loc, ti_h_loc)
    loss = data_residual_loss(ComplexF32, 2, 2)
    # calibration_options = calibration(:readout, 1f10)
    # calibration_options = calibration(:global, 0f0)
    calibration_options = nothing
    opt_parest = parameter_estimation_options(Float32; loss=loss, niter=niter_parest[i], steplength=1f0, λ=0f0, cdiag=1f-3, cid=1f-6, reg_matrix=nothing, interp_matrix=Ip_c2f_loc, calibration=calibration_options, verbose=true)
    ## Image reconstruction
    # struct_prior ? (η = 1f-1*structural_mean(prior_h); P = structural_weight(prior_h; η=η)) : (P = nothing)
    struct_prior ? (η = 1f-2; P = structural_weight(prior_h; η=η)) : (P = nothing)
    g = gradient_norm(2, 1, size(u_conventional_h), (1f0,1f0,1f0); weight=P, T=ComplexF32)
    ε = 1f0
    opt_proj = opt_fista(1f0/12f0; niter=5, Nesterov=true)
    prox(u, _) = project(u, ε, g, opt_proj)
    opt_recon = image_reconstruction_FISTA_options(Float32; loss=loss, niter=niter_imrecon[i], steplength=nothing, niter_EstLipschitzConst=3, prox=prox, Nesterov=true, calibration=calibration_options, verbose=true)
    # opt_recon = image_reconstruction_FISTA_options(Float32; loss=loss, niter=niter_imrecon[i], steplength=nothing, niter_EstLipschitzConst=3, prox=prox, Nesterov=true, calibration=nothing, verbose=true)
    ## Global
    opt = motion_correction_options(; niter=niter_outloop[i], image_reconstruction_options=opt_recon, parameter_estimation_options=opt_parest, verbose=true)

    # Selecting motion parameters on low-dimensional space
    idx_θ_h = downscale_phase_encode_index(K; fact=scale)
    # t_glob = Float32.(1:nt)
    # t_h_glob = Float32.(idx_θ_h)
    # Ip_g2h = interpolation1d_motionpars_linop(t_glob, t_h_glob)
    # Ip_h2g = interpolation1d_motionpars_linop(t_h_glob, t_glob)
    θ_h = Ip_f2c_loc*vec(θ[idx_θ_h, :])

    # Joint reconstruction
    println("\nScale = ", scale, ", regularization = ", ε_rel, "\n")
    # (scale == 0) ? (ε = 0.5f0*g(u_conventional_h)) : (ε = ε_rel*g(u_conventional_h))
    ε = ε_rel*g(u_conventional_h)
    opt.image_reconstruction.prox = (u, _) -> project(u, ε, g, opt_proj)
    u, θ_h, fval = motion_corrected_reconstruction(F_h, data_h, u, θ_h, opt)

    # Up-scaling results
    # θ .= reshape(Ip_h2g*(Ip_c2f_loc*θ_h), nt, 6)
    θ[idx_θ_h, :] .= reshape(Ip_c2f_loc*θ_h, nt_h, 6)
    θ .= envelope(θ; window=1000)
    # opt_proj_θ = opt_fista(1f0/8f0; niter=50, Nesterov=true)
    # g_θ = gradient_norm(2, 1, X.n[1:2], (1f0,1f0); T=Float32)
    # for i = 1:6
    #     θi = reshape(θ[:, i], X.n[1:2])
    #     ε_θ = 0.8f0*g_θ(θi)
    #     θ[:, i] .= vec(project(θi, ε_θ, g_θ, opt_proj_θ))
    # end
    # θ .= reshape(Ip_h2g*(Ip_h2g\vec(θ)), :, 6)./reshape([norm(θ[:,1],Inf) norm(θ[:,2],Inf) norm(θ[:,3],Inf) norm(θ[:,4],Inf) norm(θ[:,5],Inf) norm(θ[:,6],Inf)], 1, 6)
    u = upscale(u, F_h.X; fact=n_scales-i+1)

    # Plot
    plot_3D_result(u, 0, maximum(abs.(u)); x=121, y=121, z=121, filepath=string(figures_folder, "temp_e", string(ε_rel)), ext=".png")
    close("all")

end

# # Final rigid registration wrt ground-truth
# opt_reg = rigid_registration_options(Float32; niter=10, steplength=1f0, calibration=true, verbose=true)
# u, _, _ = rigid_registration(u, ground_truth, nothing, opt_reg)

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
# plot_3D_result(ground_truth, vmin, vmax; x=x, y=y, z=z, filepath=string(figures_folder, "ground_truth"), ext=".png")
# plot_3D_result(u_conventional, vmin, vmax; x=x, y=y, z=z, filepath=string(figures_folder, "conventional"), ext=".png")
# plot_3D_result(u, vmin, vmax; x=x, y=y, z=z, filepath=string(figures_folder, extra), ext=".png")
# plot_parameters(1:size(θ,1), θ, nothing; fmt1="b", fmt2="r--", linewidth1=2, linewidth2=1, filepath=string(figures_folder, "motion_pars_", extra), ext=".png")