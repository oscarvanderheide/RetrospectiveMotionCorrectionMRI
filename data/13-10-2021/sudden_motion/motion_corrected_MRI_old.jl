using BlindMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, AbstractLinearOperators, PyPlot, JLD

# Experiment name/type
phantom_id = "Invivo3D_13-10-2021"
motion_id  = "sudden_motion"
println(string(phantom_id, ": ", motion_id))

# Setting folder/savefiles
phantom_folder = string(pwd(), "/data/", phantom_id, "/")
data_folder    = string(pwd(), "/data/", phantom_id, "/", motion_id, "/")
results_folder = string(pwd(), "/data/", phantom_id, "/", motion_id, "/results/")
figures_folder = string(pwd(), "/data/", phantom_id, "/", motion_id, "/figures/")
~isdir(results_folder) && mkdir(results_folder)
~isdir(figures_folder) && mkdir(figures_folder)

# Loading data & sampling settings
X    = load(string(data_folder, "data.jld"))["X"]
K    = load(string(data_folder, "data.jld"))["K"]
data = load(string(data_folder, "data.jld"))["data"]

# Set structure-guided prior
struct_prior = true
# struct_prior = false
struct_prior && (prior = load(string(phantom_folder, "prior.jld"))["prior"])

# Setting Fourier operator
F = nfft(X, K; tol=1f-6)
nt, nk = size(K)
u_conventional = F(zeros(Float32,nt,6))'*data

# Multi-scale inversion
u = nothing
θ = zeros(Float32, nt, 6)
scale_max = 2
niter_imrecon = [10, 10, 10]
niter_parest  = [1, 1, 1]
niter_outloop = [10, 10, 10]
for i = 1:2#scale_max+1
    global u

    # Down-scaling the problem...
    scale = scale_max-i+1
    println("Scale = ", scale)
    F_h              = downscale(F;                 fact=scale)
    nt_h, _ = size(F_h.K)
    data_h           = downscale(data,           K; fact=scale)
    u_conventional_h = F_h(zeros(Float32, nt_h, 6))'*data_h
    struct_prior && (prior_h = downscale(prior, X; fact=scale))

    # Optimization options ###

    ## Weighting data misfit
    # norm_d = sqrt.(sum(abs.(data_h).^2; dims=2)); norm_d .+= 1f-3*sum(norm_d)/size(norm_d,1)
    # # W = linear_operator(ComplexF32, size(data_h), size(data_h), r->r./norm_d, r->r./norm_d)
    # W = linear_operator(ComplexF32, size(data_h), size(data_h), r->r.*norm_d, r->r.*norm_d)
    W = nothing

    ## Image reconstruction
    η = 1f0*structural_mean(prior_h)
    struct_prior ? (P = structural_weight(prior_h; η=η)) : (P = nothing)
    g = gradient_norm(2, 1, size(u_conventional_h), (1f0,1f0,1f0); weight=P, T=ComplexF32)
    (scale != 0) ? (ε = 0.3f0*g(u_conventional_h)) : (ε = 0.3f0*g(u_conventional_h))
    opt_proj = opt_fista(1f0/12f0; niter=10, Nesterov=true)
    prox(u, _) = project(u, ε, g, opt_proj)
    opt_recon = image_reconstruction_FISTA_options(; niter=niter_imrecon[i], steplength=nothing, niter_step_est=10, prox=prox, W=W, Nesterov=true, verbose=true)

    ## Parameter estimation
    # ti = Float32.(1:nt_h)
    ti = Float32.(range(1, nt_h; length=size(prior_h,1)))
    t  = Float32.(1:nt_h)
    Ip = interpolation1d_motionpars_linop(ti, t)
    D = derivative1d_motionpars_linop(t, 2; pars=(true, true, true, true, true, true))/4f0
    opt_parest = parameter_estimation_options(; niter=niter_parest[i], steplength=1f0, λ=0f0, cdiag=1f-5, cid=1f-1, reg_matrix=D, interp_matrix=Ip, W=W, verbose=true)

    ## Global
    opt = motion_correction_options(; niter=niter_outloop[i], image_reconstruction_options=opt_recon, parameter_estimation_options=opt_parest, verbose=true)

    ### End optimization options #

    # Projecting motion parameters on low-dimensional space
    idx_θ_h = downscale_phase_encode_index(K; fact=scale)
    θ_h = Float32.(Ip\vec(θ[idx_θ_h, :]))

    # Joint reconstruction
    u, θ_h, fval = motion_corrected_reconstruction(F_h, data_h, u, θ_h, opt)
    θ[idx_θ_h, :] .= reshape(Ip*vec(θ_h), nt_h, 6)

    # Up-scaling results
    (scale != 0) && (u = upscale(u, F_h.X; fact=1))

end

# # Reconstruction quality
# ground_truth = load(string(phantom_folder, "ground_truth.jld"))["ground_truth"]
# ssim_recon = ssim(u, ground_truth)
# psnr_recon = psnr(u, ground_truth)
# println("Joint reconstruction: ssim_recon = ", ssim_recon, ", psnr_recon = ", psnr_recon)

# # Save and plot results
# include(string(pwd(), "/scripts/plot_results.jl"))
# x, y, z = div.(size(ground_truth),2).+1
# vmin = minimum(abs.(ground_truth)); vmax = maximum(abs.(ground_truth))
# struct_prior ? (extra = "joint_sTV") : (extra = "joint_TV")
# struct_prior && (@save string(results_folder, "results_", extra, ".jld") u θ ssim_recon psnr_recon)
# plot_3D_result(prior, minimum(abs.(prior)), maximum(abs.(prior)); x=x, y=y, z=z, filepath=string(figures_folder, "prior"), ext=".png")
# plot_3D_result(ground_truth, vmin, vmax; x=x, y=y, z=z, filepath=string(figures_folder, "ground_truth"), ext=".png")
# plot_3D_result(u_conventional, vmin, vmax; x=x, y=y, z=z, filepath=string(figures_folder, "conventional"), ext=".png")
# plot_3D_result(u, vmin, vmax; x=x, y=y, z=z, filepath=string(figures_folder, extra), ext=".png")
# θ_true = load(string(data_folder, "motion_ground_truth.jld"))["θ"]
# plot_parameters(1:size(θ,1), θ, θ_true; fmt1="b", fmt2="r--", linewidth1=2, linewidth2=1, filepath=string(figures_folder, "motion_pars_", extra), ext=".png")