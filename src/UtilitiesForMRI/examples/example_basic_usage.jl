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
F = nfft_linop(X, K)

# Rigid-body perturbation
θ = zeros(Float64, nt, 6)
θ[:, 1] .= 0       # x translation
θ[:, 2] .= 0       # y translation
θ[:, 3] .= 0       # z translation
θ[:, 4] .= 2*pi/10 # xy rotation
θ[:, 5] .= 0       # xz rotation
θ[:, 6] .= 0       # yz rotation

# 3D image
u = zeros(ComplexF64, n); u[33-10:33+10, 33-10:33+10, 33-10:33+10] .= 1

# Rigid-body motion
u_rbm = F'*F(θ)*u

# Plotting
figure()
imshow(abs.(u_rbm[:,:,33]); vmin=0, vmax=1)