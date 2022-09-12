using LinearAlgebra, BlindMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Experiment name/type
experiment_name = "Invivo3D_17-06-2022"

# Setting folder/savefiles
data_folder = string(pwd(), "/data/", experiment_name, "/")

# Loading data
ground_truth = load(string(data_folder, "ground_truth.jld"))["ground_truth"]
X = load(string(data_folder, "data.jld"))["X"]
K = load(string(data_folder, "data.jld"))["K"]
data = load(string(data_folder, "data.jld"))["data"]

# Setting Fourier operator
F = nfft(X, K; tol=1f-6)

# Parameter estimation options
nt, _ = size(F.K)
ti = Float32.(range(1, nt; length=size(ground_truth,1)))
# ti = Float32.(1:nt)
t  = Float32.(1:nt)
Ip = interpolation1d_motionpars_linop(ti, t)
loss = data_residual_loss(ComplexF32, 2, 2)
# calibration_options = calibration(:readout, 1f10)
calibration_options = nothing
opt_parest = parameter_estimation_options(Float32; loss=loss, niter=50, steplength=1f0, λ=0, cdiag=1f-13, cid=1f10, reg_matrix=nothing, interp_matrix=Ip, calibration=calibration_options, verbose=true)

# Parameter estimation
θ0 = zeros(Float32, size(Ip,2))
# θ0 = zeros(Float32, nt, 6)
θ, fval = parameter_estimation(F, ground_truth, data, θ0, opt_parest)

# # Plot results
θ = reshape(Ip*θ, nt, 6)
u_conventional = F(0*θ)'*data
u = F(θ)'*data
include(string(pwd(), "/scripts/plot_results.jl"))
figures_folder = string(data_folder, "figures/")
~isdir(figures_folder) && mkdir(figures_folder)
x, y, z = div.(size(ground_truth),2).+1
vmin = 0; vmax = maximum(abs.(ground_truth))
# plot_3D_result(ground_truth, vmin, vmax; h=K.X.h, x=x, y=y, z=z, filepath=string(figures_folder, "ground_truth_check"), ext=".png")
# plot_3D_result(u_conventional, vmin, vmax; h=K.X.h, x=x, y=y, z=z, filepath=string(figures_folder, "conventional_check"), ext=".png", aspect="auto")
# plot_3D_result(u, vmin, vmax; h=K.X.h, x=x, y=y, z=z, filepath=string(figures_folder, "reconstructed_check"), ext=".png", aspect="auto")