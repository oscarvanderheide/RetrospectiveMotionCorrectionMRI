using LinearAlgebra, AbstractLinearOperators, AbstractProximableFunctions, FFTW, PyPlot

# Setting noisy data (plane wave)
n = (256, 256)   # Image size
fov = (1f0, 1f0) # Field-of-view
h = fov./n       # Spacing
x = range(-fov[1]/2, fov[1]/2; length=n[1]); y = range(-fov[2]/2, fov[2]/2; length=n[2]); # Cartesian coordinates
k = (0.01f0, 0.02f0)./(2f0.*h) # Wavenumber
x_clean = exp.(1im*2*Float32(pi)*(k[1]*reshape(x, :, 1).+k[2]*reshape(y, 1, :))) # Plane wave
x_noisy = x_clean.+0.5f0*randn(ComplexF32, n) # Adding noise

# Initializing regularization functionals (w/out transform)
g1   = norm(ComplexF32, 2, 1)   # ℓ1 norm for 2D (complex) images
g2   = norm(ComplexF32, 2, 2)   # ℓ2 norm for 2D (complex) images
gInf = norm(ComplexF32, 2, Inf) # ℓInf norm for 2D (complex) images

# Denoising (w/out transform)
λ1 = 0.5f0*norm(x_clean-x_noisy)^2/g1(x_clean)
x1 = prox(x_noisy, λ1, g1)
λ2 = 0.5f0*norm(x_clean-x_noisy)^2/g2(x_clean)
x2 = prox(x_noisy, λ2, g2)
λInf = 0.5f0*norm(x_clean-x_noisy)^2/gInf(x_clean)
xInf = prox(x_noisy, λInf, gInf)

# Set Fourier transform as a linear operator (wrapper via AbstractLinearOperators)
C = Float32(sqrt(prod(n)))
F = linear_operator(ComplexF32, n, n, x -> ifftshift(fft(x))/C, # Forward evaluation
                                      x̂ -> C*ifft(fftshift(x̂))) # Adjoint evaluation

# Transform-based regularization functionals
g1_F = g1∘F
g2_F = g2∘F
gInf_F = gInf∘F

# Minimization options
options = FISTA_options(1f0; niter=10)

# Denoising (w/ Fourier transform)
λ1_F = 0.5f0*norm(x_clean-x_noisy)^2/g1_F(x_clean)
x1_F = prox(x_noisy, λ1_F, g1_F, options)
λ2_F = 0.5f0*norm(x_clean-x_noisy)^2/g2_F(x_clean)
x2_F = prox(x_noisy, λ2_F, g2_F, options)
λInf_F = 0.5f0*norm(x_clean-x_noisy)^2/gInf_F(x_clean)
xInf_F = prox(x_noisy, λInf_F, gInf_F, options)

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