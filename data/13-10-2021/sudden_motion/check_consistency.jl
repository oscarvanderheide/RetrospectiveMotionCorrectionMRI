using BlindMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, JLD

# Experiment name/type
phantom_id = "Invivo3D_13-10-2021"; motion_id = "sudden_motion"
experiment_name = string(phantom_id, "/", motion_id)

# Setting folder/savefiles
phantom_folder = string(pwd(), "/data/", phantom_id, "/")
data_folder    = string(pwd(), "/data/", experiment_name, "/")

# Loading data
ground_truth = load(string(phantom_folder, "ground_truth.jld"))["ground_truth"]
prior = load(string(phantom_folder, "prior.jld"))["prior"]
X = load(string(data_folder, "data.jld"))["X"]
K = load(string(data_folder, "data.jld"))["K"]
data = load(string(data_folder, "data.jld"))["data"]

# Setting Fourier operator
F = nfft(X, K; tol=1f-5)

# Downscale
fact = 1
F = downscale(F; fact=fact)
data = downscale(data, K; fact=fact)
ground_truth = downscale(ground_truth, X; fact=fact)
K = F.K
nt, nk = size(K)
# u_conventional = F(zeros(Float32,nt,6))'*data

# Parameter estimation options
nt, _ = size(F.K)
# ti = Float32.(1:nt)
ti = Float32.(range(1, nt; length=F.X.n[1]))
t  = Float32.(1:nt)
Ip = interpolation1d_motionpars_linop(ti, t)
D = derivative1d_motionpars_linop(t, 2; pars=(true, true, true, true, true, true))/4f0
opt_parest = parameter_estimation_options(; niter=20, steplength=1f0, λ=0f0, cdiag=1f-5, cid=1f-1, reg_matrix=D, interp_matrix=Ip, W=nothing, verbose=true)

# Parameter estimation
θ0 = zeros(Float32, length(ti)*6)
θ, fval = parameter_estimation(F, ground_truth, data, θ0, opt_parest)
θ = reshape(Ip*θ, nt, 6)

# Plot results
u_conventional = F(0*θ)'*data
u = F(θ)'*data
include(string(pwd(), "/scripts/plot_results.jl"))
figures_folder = string(pwd(), "/data/", phantom_id, "/", motion_id, "/figures/")
~isdir(figures_folder) && mkdir(figures_folder)
x, y, z = div.(size(ground_truth),2).+1
vmin = minimum(abs.(ground_truth)); vmax = maximum(abs.(ground_truth))
plot_3D_result(ground_truth, vmin, vmax; h=K.X.h, x=x, y=y, z=z, filepath=string(figures_folder, "ground_truth_check"), ext=".png")
plot_3D_result(u_conventional, vmin, vmax; h=K.X.h, x=x, y=y, z=z, filepath=string(figures_folder, "conventional_check"), ext=".png")
plot_3D_result(u, vmin, vmax; h=K.X.h, x=x, y=y, z=z, filepath=string(figures_folder, "reconstructed_check"), ext=".png")