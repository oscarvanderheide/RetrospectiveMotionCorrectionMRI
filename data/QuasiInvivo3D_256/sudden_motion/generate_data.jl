using LinearAlgebra, BlindMotionCorrectionMRI, UtilitiesForMRI, PyPlot, JLD

# Setting folder/savefiles
phantom_id = "QuasiInvivo3D_256"
motion_id  = "sudden_motion"
data_folder = string(pwd(), "/data/", phantom_id, "/")

# Loading data
h = [1f0, 1f0, 1f0]
ground_truth = ComplexF32.(load(string(data_folder, "ground_truth.jld"))["ground_truth"])
n = size(ground_truth)

# Setting simulated motion
θ = zeros(Float32, n[1:2]..., 6)
θ[:, div(n[2],2)+3:end, 1] .= -5
θ[:, div(n[2],2)+3:end, 4] .= 10f0*pi/180
θ = reshape(θ, :, 6)

# Generating synthetic data
X = spatial_sampling(n; h=h)
K = kspace_Cartesian_sampling(X; phase_encoding=(1,2))
F = nfft(X, K; tol=1f-5)
Fθ = F(θ)
snr = 90f0 # dB
data = Fθ*ground_truth
data .+= 10^(-snr/20f0)*norm(data)*randn(ComplexF32, size(data))
# u_conventional = nfft_linop(X, K; tol=1f-5)'*data

# Saving data
save(string(data_folder, motion_id, "/data.jld"), "data", data, "X", X, "K", K, "snr", snr)
save(string(data_folder, motion_id, "/motion_ground_truth.jld"), "θ", θ)