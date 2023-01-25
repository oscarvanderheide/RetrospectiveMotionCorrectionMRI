using LinearAlgebra, RetrospectiveMotionCorrectionMRI, AbstractProximableFunctions, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Folders
experiment_name = "hybrid-modalities"; @info string(experiment_name, "\n")
exp_folder = string(pwd(), "/data/", experiment_name, "/")
data_folder = string(exp_folder, "data/")
~isdir(data_folder) && mkdir(data_folder)
figures_folder = string(exp_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)
results_folder = string(exp_folder, "results/")
~isdir(results_folder) && mkdir(results_folder)

# Volunteer, reconstruction type (custom vs DICOM), and motion type
experiment_subname = "vol1_priorT1"; motion_type = 3

# Loading data
@info string("Exp: ", experiment_subname, ", motion type: ", motion_type)
figures_subfolder = string(figures_folder, experiment_subname, "/"); ~isdir(figures_subfolder) && mkdir(figures_subfolder)
data_file = string("data_", experiment_subname, ".jld")
X, K, data, prior, ground_truth, corrupted, orientation, mask = load(string(data_folder, data_file), "X", "K", string("data_motion", motion_type), "prior", "ground_truth", string("corrupted_motion", motion_type), "orientation", "mask")

# Setting Fourier operator
F = nfft_linop(X, K)
nt, nk = size(K)

# Setting starting values
θ = zeros(Float32, nt, 6)

# Down-scaling the problem (temporally)...
nt_h = 64
t_coarse = Float32.(range(1, nt; length=nt_h))
t_fine = Float32.(1:nt)
# interp = :spline
interp = :linear
Ip_c2f = interpolation1d_motionpars_linop(t_coarse, t_fine; interp=interp)
Ip_f2c = interpolation1d_motionpars_linop(t_fine, t_coarse; interp=interp)

## Parameter estimation options
scaling_diagonal = 1f-3
scaling_mean     = 1f-4
scaling_id       = 0f0
Ip_c2f = interpolation1d_motionpars_linop(t_coarse, Float32.(t_fine); interp=interp)
D = derivative1d_motionpars_linop(t_coarse, 1; pars=(true, true, true, true, true, true))
λ = 1f-3*sqrt(norm(data)^2/spectral_radius(D'*D; niter=3))
opt_parest = parameter_estimation_options(; niter=100, steplength=1f0, λ=λ, scaling_diagonal=scaling_diagonal, scaling_mean=scaling_mean, scaling_id=scaling_id, reg_matrix=D, interp_matrix=Ip_c2f, calibration=true, verbose=true)

# Selecting motion parameters on low-dimensional space
θ_coarse = reshape(Ip_f2c*vec(θ), :, 6)

# Estimation
θ_coarse = parameter_estimation(F, ground_truth, data, θ_coarse, opt_parest)

# Up-scaling motion parameters
θ .= reshape(Ip_c2f*vec(θ_coarse), :, 6)

# Plot
θ_min =  minimum(θ; dims=1); θ_max = maximum(θ; dims=1); Δθ = θ_max-θ_min; θ_middle = (θ_min+θ_max)/2
Δθ = [ones(1,3)*max(Δθ[1:3]...)/2 ones(1,3)*max(Δθ[4:end]...)/2]
plot_parameters(1:nt, θ, nothing; xlabel="t = phase encoding", vmin=vec(θ_middle-1.1f0*Δθ), vmax=vec(θ_middle+1.1f0*Δθ), fmt1="b", linewidth1=2, savefile=string(figures_subfolder, "temp_motion_pars.png"))
close("all")