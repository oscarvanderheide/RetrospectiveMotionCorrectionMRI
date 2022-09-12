using LinearAlgebra, RetrospectiveMotionCorrectionMRI, ConvexOptimizationUtils, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Experiment name/type
experiment_name = "17-06-2022_SENSE"

# Setting folder/savefiles
data_folder = string(pwd(), "/data/", experiment_name, "/")

# Loading data
ground_truth = load(string(data_folder, "ground_truth.jld"))["ground_truth"]
X = load(string(data_folder, "data.jld"))["X"]
K = load(string(data_folder, "data.jld"))["K"]
data = load(string(data_folder, "data.jld"))["data"]

# Setting Fourier operator
F = nfft_linop(X, K)

# Parameter estimation options
nt, _, _ = size(F.kcoord)
ti = Float32.(range(1, nt; length=size(ground_truth,1)))
# ti = Float32.(1:nt)
t  = Float32.(1:nt)
Ip = interpolation1d_motionpars_linop(ti, t)
D = derivative1d_motionpars_linop(t, 2; pars=(true, true, true, true, true, true))/4f0
opt_parest = parameter_estimation_options(; niter=100, steplength=1f0, λ=0f0, cdiag=1f-5, cid=1f-1, reg_matrix=D, interp_matrix=Ip, verbose=true, fun_history=true)

# Parameter estimation
θ0 = zeros(Float32, size(Ip,2))
θ = parameter_estimation(F, ground_truth, data, θ0, opt_parest)

# Plot results
θ = reshape(Ip*θ, nt, 6)
u_conventional = F(0*θ)'*data
u = F(θ)'*data
figures_folder = string(data_folder, "figures/"); ~isdir(figures_folder) && mkdir(figures_folder)
x, y, z = div.(size(ground_truth),2).+1
vmin = 0; vmax = maximum(abs.(ground_truth))
slices = (VolumeSlice(1, x), VolumeSlice(2, y), VolumeSlice(3, z))
plot_volume_slices(abs.(u); slices=slices, spatial_geometry=X, vmin=vmin, vmax=vmax, savefile=string(figures_folder, "reconstructed_check.png"))