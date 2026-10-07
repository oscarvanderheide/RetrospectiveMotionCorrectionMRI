using UtilitiesForMRI, LinearAlgebra, PythonPlot

# Cartesian domain
n = (64, 64, 64)
fov = (1.0, 1.0, 1.0)
origin = (0.5, 0.5, 0.5)
X = spatial_geometry(fov, n; origin=origin)

# Cartesian sampling in k-space
phase_encoding_dims = (1,2)
K = kspace_sampling(X, phase_encoding_dims)
nt, nk = size(K)

# Fourier operator
tol = 1e-6
F = nfft_linop(X, K; tol=tol)

# Rigid-body perturbation
θ = zeros(Float64, nt, 6)

# 3D image
u = zeros(ComplexF64, n); u[33-10:33+10, 33-10:33+10, 33-10:33+10] .= 1

# Rigid-body motion
for θxy = range(0,2*pi; length=20)
    θ[:, 4] .= θxy
    u_rbm = F'*F(θ)*u
    imshow(abs.(u_rbm[:,:,33]); vmin=0, vmax=1)
    pause(0.1)
end