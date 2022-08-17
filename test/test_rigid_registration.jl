using RetrospectiveMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, ConvexOptimizationUtils, LinearAlgebra, Test

# Spatial geometry
fov = (1f0, 2f0, 2f0)
n = (64, 64, 64)
o = (0.5f0, 1f0, 1f0)
X = spatial_geometry(fov, n; origin=o)

# Cartesian sampling in k-space
phase_encoding = (1,2)
K = kspace_sampling(X, phase_encoding)
nt, nk = size(K)

# Fourier operator
F = nfft_linop(X, K)

# Setting ground-truth and data
ground_truth = zeros(ComplexF32, n); ground_truth[div(64,2)+1-10:div(64,2)+1+10, div(64,2)+1-10:div(64,2)+1+10, div(64,2)+1-10:div(64,2)+1+10] .= 1
nt, _ = size(K)
θ = zeros(Float32, nt, 6)
θ .= reshape([0.0f0, 0.1f0, 0.0f0, Float32(pi)/180*10, 0f0, 0f0], 1, 6)
u = F'*(F(θ)*ground_truth)

# Optimization options
opt = rigid_registration_options(; T=Float32, niter=100, verbose=true, fun_history=true)

# Solution
u_ = rigid_registration(ground_truth, u, nothing, opt; spatial_geometry=X)