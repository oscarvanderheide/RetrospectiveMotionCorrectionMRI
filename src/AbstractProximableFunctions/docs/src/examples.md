# Getting started

We show some of the functionalities of `AbstractProximableFunctions` with a simple tutorial, focus on a 2D denoising problem. Before starting, make sure the package is installed (see Section [Installation instructions](@ref)). For this tutorial, we also need `FFTW` (Fourier transform), `TestImages`, `PyPlot`. To install,
```julia
@(v1.8) pkg> add FFTW, TestImages, PyPlot
```
Here, we use `PyPlot` for image visualization, but many other packages may fit the bill.

To load all the needed packages:
```julia
using LinearAlgebra, AbstractLinearOperators, AbstractProximableFunctions, FFTW, PyPlot
```
For our denoising exercise, we initialize a noisy image, which consists of a simple complex plane wave:
```julia
# Setting noisy data (plane wave)
n = (256, 256)   # Image size
fov = (1f0, 1f0) # Field-of-view
h = fov./n       # Spacing
x = range(-fov[1]/2, fov[1]/2; length=n[1]); y = range(-fov[2]/2, fov[2]/2; length=n[2]); # Cartesian coordinates
k = (0.01f0, 0.02f0)./(2f0.*h) # Wavenumber
x_clean = exp.(1im*2*Float32(pi)*(k[1]*reshape(x, :, 1).+k[2]*reshape(y, 1, :))) # Plane wave
x_noisy = x_clean.+0.5f0*randn(ComplexF32, n) # Adding noise
```
We study the effect of three regularization functionals: ``\ell^1``, ``\ell^2``, and ``\ell^{\infty}``:
```julia
# Initializing regularization functionals (w/out transform)
g1   = norm(ComplexF32, 2, 1)   # ℓ1 norm for 2D (complex) images
g2   = norm(ComplexF32, 2, 2)   # ℓ2 norm for 2D (complex) images
gInf = norm(ComplexF32, 2, Inf) # ℓInf norm for 2D (complex) images
```
To perform denoising, e.g. find the solution of the optimization problem ``\min_{\mathbf{x}}\frac{1}{2}||\mathbf{x}-\mathbf{y}||^2+\lambda g(\mathbf{x})``, we need to set a weight ``\lambda`` and make a call to proximal operator:
```julia
# Denoising
λ1 = 0.5f0*norm(x_clean-x_noisy)^2/g1(x_clean)
x1 = prox(x_noisy, λ1, g1)
λ2 = 0.5f0*norm(x_clean-x_noisy)^2/g2(x_clean)
x2 = prox(x_noisy, λ2, g2)
λInf = 0.5f0*norm(x_clean-x_noisy)^2/gInf(x_clean)
xInf = prox(x_noisy, λInf, gInf)
```
The results of the denoising problem will not look particularly good, because we are not exploiting enough prior knowledge of the solution.

A more sensible way to go about relies on the sparsity of the solution in some domain. For this purpose, we define the Fourier transform:
```julia
# Set Fourier transform as a linear operator (wrapper via AbstractLinearOperators)
C = Float32(sqrt(prod(n)))
F = linear_operator(ComplexF32, n, n, x -> ifftshift(fft(x))/C, # Forward evaluation
                                      x̂ -> C*ifft(fftshift(x̂))) # Adjoint evaluation
```
We can compose the Fourier transform with the previous regularization functionals:
```julia
# Transform-based regularization functionals
g1_F = g1∘F
g2_F = g2∘F
gInf_F = gInf∘F
```
Performing denoising with weighted regularization is tantamount to solve the optimization problem

``
\min_{\mathbf{x}}\dfrac{1}{2}||\mathbf{x}-\mathbf{y}||^2+\lambda g(A\mathbf{x})
``

where ``A`` is a linear operator. By default, in `AbstractProximalFunctions` this problem is reformulated in dual form:

``
\min_{\mathbf{p}}\dfrac{1}{2}||\lambda A^*\mathbf{p}-\mathbf{y}||^2+\lambda g^*(\mathbf{p})
``

where ``g^*`` is the convex conjugate of ``g``. Primal ``\mathbf{x}`` and dual variables ``\mathbf{p}`` are related by ``\mathbf{x}=\mathbf{y}-\lambda A^*\mathbf{p}``.

In order to solve this problem, we resort to a FISTA iterative solver. To specify FISTA, we have to know the Lipschitz constant of ``\nabla f`` where ``f(\mathbf{x})=\dfrac{1}{2}||\lambda A^*\mathbf{p}-\mathbf{y}||^2``, which is ``\mathrm{Lip}\nabla f=\lambda^2\rho(A)`` with ``\rho`` being the spectral radius. Importantly, whenever computing the `prox` of `WeighthedProximalFunction`s, `FISTA_options` actually expects ``\mathrm{Lip}\nabla f=\rho(A)``. Hence:
```julia
# Minimization options
options = FISTA_options(1f0;           # Lipschitz constant of ∇f(x), in this case equivalent to spectral norm of F
                             niter=10) # Number of iteration
```
Now, everything is set up to run the proximal computations:
```julia
# Denoising (w/ Fourier transform)
λ1_F = 0.5f0*norm(x_clean-x_noisy)^2/g1_F(x_clean)
x1_F = prox(x_noisy, λ1_F, g1_F, options)
λ2_F = 0.5f0*norm(x_clean-x_noisy)^2/g2_F(x_clean)
x2_F = prox(x_noisy, λ2_F, g2_F, options)
λInf_F = 0.5f0*norm(x_clean-x_noisy)^2/gInf_F(x_clean)
xInf_F = prox(x_noisy, λInf_F, gInf_F, options)
```

We expect the transform-based ``\ell^1`` to performed the best. In order to compare the results:
```julia
# Plot
vmin = -1f0
vmax =  1f0
figure()
subplot(2,5,1)
imshow(real(x_noisy); vmin=vmin, vmax=vmax)
title("Noisy phantom")
axis("off")
subplot(2,5,2)
imshow(real(xInf); vmin=vmin, vmax=vmax)
title(L"$\ell^{\infty}$ denoising")
axis("off")
subplot(2,5,3)
imshow(real(x2); vmin=vmin, vmax=vmax)
title(L"$\ell^2$ denoising")
axis("off")
subplot(2,5,4)
imshow(real(x1); vmin=vmin, vmax=vmax)
title(L"$\ell^1$ denoising")
axis("off")
subplot(2,5,5)
imshow(real(x_clean); vmin=vmin, vmax=vmax)
title("Ground truth")
axis("off")
subplot(2,5,6)
imshow(real(x_noisy); vmin=vmin, vmax=vmax)
title("Noisy phantom")
axis("off")
subplot(2,5,7)
imshow(real(xInf_F); vmin=vmin, vmax=vmax)
title(L"$\ell^{\infty}$ denoising (w/ transform)")
axis("off")
subplot(2,5,8)
imshow(real(x2_F); vmin=vmin, vmax=vmax)
title(L"$\ell^2$ denoising (w/ transform)")
axis("off")
subplot(2,5,9)
imshow(real(x1_F); vmin=vmin, vmax=vmax)
title(L"$\ell^1$ denoising (w/ transform)")
axis("off")
subplot(2,5,10)
imshow(real(x_clean); vmin=vmin, vmax=vmax)
title("Ground truth")
axis("off")
```