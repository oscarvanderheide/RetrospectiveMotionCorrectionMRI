using LinearAlgebra, BlindMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, Flux, PyPlot, JLD, ImageFiltering, Images
using Random; Random.seed!(123)
include("./scripts/plot_results.jl")

# Setting folder and savefiles
# phantom_id = "SheppLogan3D"
phantom_id = "BrainWeb3D_128"
motion_id  = "rnd"
# motion_id  = "sudden"
data_folder    = string("./data/",    phantom_id, "/")
results_folder = string("./results/", phantom_id, "/")
figures_folder = string("./figures/", phantom_id, "/")
experiment_name = string(phantom_id, "_", motion_id)
results_filename(extra) = string("results_", experiment_name, "_", extra,".jld")

# Loading numerical phantom
println(experiment_name)
h = [1f0, 1f0, 1f0]
u_true = Float32.(load(string(data_folder, phantom_id, ".jld"))["T2"])
prior = Float32.(load(string(data_folder, phantom_id, ".jld"))["T1"])
n = size(u_true)
u_true = complex(u_true/norm(u_true, Inf))
prior = complex(prior/norm(prior, Inf))
X = spatial_sampling(n; h=h)
K = kspace_sampling(X; readout=:z, phase_encode=:xy)
nt, nk = size(K)

# Setting simulated motion
if motion_id == "sudden"
    θ_true = zeros(Float32, n[1:2]..., 6)
    θ_true[:, n[2]+3:end, 1:3] .= 2f0
    θ_true[:, n[2]+3:end, 4:6] .= pi/180*2
elseif motion_id == "rnd"
    θ_true = imresize(randn(Float32, Int64(floor(nt/(5*n[1]))), 6), nt, 6)
    smooth_fact = nt/16
    w = Kernel.gaussian((smooth_fact, ))
    for i = 1:3
        θ_true[:, i] = imfilter(θ_true[:, i], w, "circular")
        θ_true[:, i] .*= 3f0/norm(θ_true[:, i], Inf)
    end
    for i = 4:6
        θ_true[:, i] = imfilter(θ_true[:, i], w, "circular")
        θ_true[:, i] .*= 3f0/(180f0*norm(θ_true[:, i], Inf))
    end
end

# Generating synthetic data
F = nfft(X, K; tol=1f-5)
Fθ_true = F(θ_true)
snr = 90f0 # dB
d = Fθ_true*u_true
d .+= 10^(-snr/20f0)*norm(d)*randn(ComplexF32, size(d))

# Conventional solution
u_conventional = nfft_linop(X, K; tol=1f-5)'*d
ssim_conventional = ssim(u_conventional, u_true)
psnr_conventional = psnr(u_conventional, u_true)
println("Conventional: SSIM = ", ssim_conventional, ", PSNR = ", psnr_conventional)

# Save and plot results
@save string(results_folder, results_filename("conventional")) u_conventional ssim_conventional psnr_conventional
vmin = minimum(abs.(u_true)); vmax = maximum(abs.(u_true))
x = Int64(floor(n[1]/2))
y = Int64(floor(n[2]/2))
z = Int64(floor(n[2]/2))
plot_3D_result(u_conventional, vmin, vmax; x=x, y=y, z=z, filepath=string(figures_folder, experiment_name, "_conventional"), ext=".png")

# Optimization options
    ## Image reconstruction
    η = 1f-0*structural_mean(prior)
    P = structural_weight(prior; η=η)
    # P = nothing
    g = gradient_norm(2, 1, n, tuple(h...); weight=P, T=ComplexF32)
    ε = 0.9f0*g(u_true)
    opt_proj = opt_fista(1f0/12f0; niter=10, Nesterov=true)
    prox(u, λ) = project(u, ε, g, opt_proj)
    opt_recon = image_reconstruction_FISTA_options(; niter=10, steplength=nothing, niter_spect=10, prox=prox, Nesterov=true, verbose=true)

    ## Parameter estimation
    ti = Float32.(range(1, nt; length=Int64(nt/64)))
    t = Float32.(1:nt)
    Ip = interpolation1d_motionpars_linop(ti, t)
    D = derivative1d_motionpars_linop(t, 2; pars=(true, true, true, true, true, true))/4f0
    opt_parest = parameter_estimation_options(; niter=1, steplength=1f0, λ=0f0, cdiag=1f-5, cid=1f-1, reg_matrix=D, interp_matrix=Ip, verbose=true)

    ## Global options
    opt = motion_correction_options(; niter=10, image_reconstruction_options=opt_recon, parameter_estimation_options=opt_parest, verbose=true) 

# Joint reconstruction
u = deepcopy(u_conventional)
θ = zeros(Float32, size(Ip, 2))
u_joint, θ_joint, fval = motion_corrected_reconstruction(F, d, u, θ, opt)
θ_joint = reshape(Ip*vec(θ_joint), nt, 6)
ssim_joint = ssim(u_joint, u_true)
psnr_joint = psnr(u_joint, u_true)
println("Joint reconstruction: SSIM = ", ssim_joint, ", PSNR = ", psnr_joint)

# Save and plot results
isnothing(P) ? (extra = "joint") : (extra = "wjoint")
@save string(results_folder, results_filename(extra)) u_joint θ_joint ssim_joint psnr_joint
plot_3D_result(u_joint, vmin, vmax; x=x, y=y, z=z, filepath=string(figures_folder, experiment_name, "_", extra), ext=".png")