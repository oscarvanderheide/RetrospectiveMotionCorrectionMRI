using RetrospectiveMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, AbstractProximableFunctions, LinearAlgebra, Test

# Spatial geometry
fov = (1f0, 2f0, 2f0)
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
h = spacing(X); LD = 4f0*sum(1 ./h.^2)
opt_inner = FISTA_options(LD; Nesterov=true, niter=20)
g = gradient_norm(2, 1, size(ground_truth), h; complex=true, options=opt_inner)
x = reshape(range(-1f0, 1f0; length=64), :, 1, 1)
y = reshape(range(-1f0, 1f0; length=64), 1, :, 1)
z = reshape(range(-1f0, 1f0; length=64), 1, 1, :)
r = sqrt.(x.^2 .+y.^2 .+z.^2)
C = zero_set(ComplexF32, r .< 0.1f0)
ε = g(ground_truth)
# h = indicator(set_options(g+indicator(C), opt_inner) ≤ ε)
h = indicator(g ≤ ε)
L = 1.1f0*spectral_radius(F'*F; niter=3)
opt = image_reconstruction_options(; prox=h, Lipschitz_constant=L, Nesterov=true, niter=30, verbose=true, fun_history=true)

# Solve
u0 = zeros(ComplexF32, size(ground_truth))
u = image_reconstruction(F, dnoise, u0, opt)

# Coherence test w/ minimize
u_ = argmin(leastsquares_misfit(F, dnoise)+h, u0, opt.options)
@test u ≈ u_