using LinearAlgebra, BlindMotionCorrectionMRI, UtilitiesForMRI,  FastSolversForWeightedTV, PyPlot, JLD, Statistics

# Setting folder/savefiles
experiment_name = "Invivo3D_21-04-2022"
data_folder = string(pwd(), "/data/", experiment_name, "/")

# Loading data
unprocessed_file = "unprocessed_scans.jld"
h = load(string(data_folder, unprocessed_file))["h"]
prior = load(string(data_folder, unprocessed_file))["T1"]
T2_motion = load(string(data_folder, unprocessed_file))["T2_motion"]
ground_truth = load(string(data_folder, unprocessed_file))["T2_nomotion"]

# # Rigid registration
# ∇ = gradient_operator(size(prior), h; T=ComplexF32)
# ∇prior = sqrt.(sum(abs.(∇*prior).^2; dims=4))[:,:,:,1]; ε = 1f1*mean(abs.(∇prior)); ∇prior .= abs.(∇prior)./sqrt.(abs.(∇prior).^2 .+ε^2)
# ∇ref = sqrt.(sum(abs.(∇*T2_motion).^2; dims=4))[:,:,:,1]; ε = 1f1*mean(abs.(∇ref)); ∇ref .= abs.(∇ref)./sqrt.(abs.(∇ref).^2 .+ε^2)

# opt_reg = rigid_registration_options(Float32; niter=150, verbose=true)
# ∇prior_, θ, _ = rigid_registration(complex(∇prior/norm(∇prior, Inf)), complex(∇ref/norm(∇ref, Inf)), nothing, opt_reg; h=h)
# n = size(T2_motion)
# X = spatial_sampling(Float32, n; h=h)
# K = kspace_Cartesian_sampling(X; phase_encoding=(3,1))
# nt, _ = size(K)
# prior = nfft_linop(X, K; tol=1f-6)'*(nfft(X, K; tol=1f-6)(repeat(θ; outer=(nt,1)))*prior)

# # Denoise prior
# g = gradient_norm(2, 1, size(prior), (1f0, 1f0, 1f0); T=ComplexF32)
# opt_proj = opt_fista(1f0/12f0; niter=20, Nesterov=true)
# prior = project(prior, 0.5f0*g(prior), g, opt_proj)
# prior ./= norm(prior, Inf)

# # Save ground-truth and prior
# save(string(data_folder, "/ground_truth.jld"), "ground_truth", ground_truth)
# save(string(data_folder, "/prior.jld"), "prior", prior)

# Generating synthetic data
n = size(T2_motion)
X = spatial_sampling(Float32, n; h=h)
K = kspace_Cartesian_sampling(X; phase_encoding=(3,1))
F = nfft_linop(X, K; tol=1f-6)
data = F*T2_motion

# Saving data
save(string(data_folder, "data.jld"), "data", data, "X", X, "K", K)