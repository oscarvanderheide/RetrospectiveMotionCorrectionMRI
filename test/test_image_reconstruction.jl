using MotionCorrectedMRI, FastSolversForWeightedTV, UtilitiesForMRI, PyPlot, LinearAlgebra

# Spatial geometry
fov = (1f0, 2f0, 3f0)
n = (64, 64, 64)
o = (0.5f0, 0.5f0, 0.5f0)
X = spatial_geometry(fov, n; origin=o)

# Cartesian sampling in k-space
phase_encoding = (1,2)
K = kspace_sampling(X, phase_encoding)
nt, nk = size(K)

# Fourier operator
tol = 1f-6
F = nfft_linop(X, K; tol=tol)

# Setting ground-truth and data
ground_truth = zeros(ComplexF32, n); ground_truth[div(64,2)+1-10:div(64,2)+1+10, div(64,2)+1-10:div(64,2)+1+10, div(64,2)+1-10:div(64,2)+1+10] .= 1
d = F*ground_truth
p = 40f0 # dB
noise = 10^(-p/20)*norm(d,Inf)*randn(ComplexF32,size(d))
dnoise = d+noise

# Image reconstruction options
h = spacing(X)
g = gradient_norm(2, 1, size(ground_truth), h; T=ComplexF32)
ε = g(ground_truth)
prox(u, _) = project(u, ε, g, opt_fista(0.25f0/sum(1f0./h.^2); niter=10, Nesterov=true))
loss = data_residual_loss(ComplexF32, 2, 2)
opt_recon = image_reconstruction_FISTA_options(loss, prox; niter=10, steplength=nothing, niter_estim_Lipschitz_const=3, Nesterov=true, reset_counter=20, verbose=true)

# u0 = zeros(ComplexF32, size(ground_truth))
u0 = 0*ground_truth
u, fval, A = image_reconstruction(F, dnoise, u0, opt_recon)