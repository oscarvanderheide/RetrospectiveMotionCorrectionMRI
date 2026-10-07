using LinearAlgebra, CUDA, TestImages, PyPlot
using AbstractProximableFunctions, FastSolversForWeightedTV

# Prepare data
n = (256, 256, 256)                                   # Image size
x_clean = Float32.(TestImages.shepp_logan(n[1:2]...)) # 2D Shepp-Logan of size 256x256
x_clean = repeat(x_clean; outer=(1,1,n[3]))           # 3D "augmentation"
x_clean = CuArray(x_clean)                            # Move data to GPU
x_clean = x_clean/norm(x_clean, Inf)                  # Normalization
x_noisy = x_clean+0.1f0*CUDA.randn(Float32, n)        # Adding noise

# Regularization
h = (1f0, 1f0, 1f0)                                                     # Grid spacing
L = 12f0                                                                # Spectral norm of the gradient operator
opt = FISTA_options(L; Nesterov=true,
                       niter=20,
                       reset_counter=10,
                       verbose=false)                                   # FISTA options
g_TV  = gradient_norm(2, 1, n, h; complex=false, options=opt) # TV

# Denoising
λ_TV = 0.5f0*norm(x_clean-x_noisy)^2/g_TV(x_clean) # Denoising weight
x_TV = prox(x_noisy, λ_TV, g_TV)                   # TV denoising

# Reference-guided regularization
η = 0.1f0*structural_mean(x_clean)                                                 # Stabilization term
P = structural_weight(x_clean; η=η)                                                # Weight based on a given reference
g_rTV  = gradient_norm(2, 1, n, h; weight=P, complex=false, options=opt) # Reference-guided TV

# Denoising (structure-guided)
λ_rTV = 0.5f0*norm(x_clean-x_noisy)^2/g_rTV(x_clean) # Denoising weight
x_rTV = prox(x_noisy, λ_rTV, g_rTV)                  # Reference-guided TV denoising

# Denoising (structure-guided projection)
ε = 0.5f0*g_rTV(x_clean)             # Noise level
x_rTV_proj = proj(x_noisy, ε, g_rTV) # Projection

# Equivalently: Denoising (structure-guided projection)
C = g_rTV ≤ ε                 # Constraint set
x_rTV_proj = proj(x_noisy, C) # Projection

# Move data back to CPU
x_clean = Array(x_clean)
x_noisy = Array(x_noisy)
x_TV = Array(x_TV)
x_rTV = Array(x_rTV)

# Plot
figure()
subplot(1, 4, 1)
title("Noisy")
imshow(abs.(x_noisy[:,:,129]); vmin=0, vmax=1, cmap="gray")
subplot(1, 4, 2)
title("TV")
imshow(abs.(x_TV[:,:,129]); vmin=0, vmax=1, cmap="gray")
subplot(1, 4, 3)
title("rTV")
imshow(abs.(x_rTV[:,:,129]); vmin=0, vmax=1, cmap="gray")
subplot(1, 4, 4)
title("Ground-truth")
imshow(abs.(x_clean[:,:,129]); vmin=0, vmax=1, cmap="gray")