# Getting started

We provide a simple example to perform a rigid-body motion using the tools provided by `UtilitiesForMRI`.

For starters, let's make sure that all the needed packages are installed! Please, follow the instructions in Section [Installation instructions](@ref). For this tutorial, we also need `PythonPlot`. To install, Type `]` in the Julia REPL and
```julia
@(v1.8) pkg> add PythonPlot
```
Here, we use `PythonPlot` for image visualization, but many other packages may fit the bill.

To load the relevant modules:
```julia
# Package load
using UtilitiesForMRI, LinearAlgebra, PythonPlot
```

Let's define a Cartesian spatial discretization for a 3D image:
```julia
# Cartesian domain
n = (64, 64, 64)
fov = (1.0, 1.0, 1.0)
origin = (0.5, 0.5, 0.5)
X = spatial_geometry(fov, n; origin=origin)
```

We can also set a simple ``k``-space trajectory:
```julia
# Cartesian sampling in k-space
phase_encoding_dims = (1,2)
K = kspace_sampling(X, phase_encoding_dims)
nt, nk = size(K)
```

The Fourier operator for 3D images based on the `X` discretization and `K` sampling is:
```julia
# Fourier operator
F = nfft_linop(X, K)
```

Let's assume we want to perform a rigid motion for a certain image:
```julia
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
```
This can be easily done by evaluating the rigid-motion perturbed Fourier transform, and applying the adjoint of the conventional Fourier transform, e.g.:
```julia
# Rigid-body motion
u_rbm = F'*F(θ)*u
```

For plotting:
```julia
# Plotting
figure()
imshow(abs.(u_rbm[:,:,33]); vmin=0, vmax=1)
```