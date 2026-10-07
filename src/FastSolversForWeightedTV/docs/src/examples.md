# Getting started

## TV vs reference-guided TV denoising

We briefly describe how to use the tools provided by this package. We focus, here, on a 3D TV-denoising example with GPU acceleration.

For starters, let's make sure that all the needed packages are installed! Please, follow the instructions in Section [Installation instructions](@ref). For this tutorial, we also need `CUDA`, `PyPlot`, and `TestImages`. To install, Type `]` in the Julia REPL and
```julia
@(v1.8) pkg> add CUDA, TestImages, PyPlot
```
Here, we use `PyPlot` for image visualization, but many other packages may fit the bill.

To load the relevant modules:
```julia
# Package load
using LinearAlgebra, CUDA, TestImages, PyPlot
using AbstractProximableFunctions, FastSolversForWeightedTV
```

Let's load the 2D Shepp-Logan phantom and make a 3D volume out of it. Also let's contaminate the volume with some random noise:
```julia
# Prepare data
n = (256, 256, 256)                                   # Image size
x_clean = Float32.(TestImages.shepp_logan(n[1:2]...)) # 2D Shepp-Logan of size 256x256
x_clean = repeat(x_clean; outer=(1,1,n[3]))           # 3D "augmentation"
x_clean = CuArray(x_clean)                            # Move data to GPU
x_clean = x_clean/norm(x_clean, Inf)                  # Normalization
x_noisy = x_clean+0.1f0*CUDA.randn(Float32, n)        # Adding noise
```

Now that we prepared the noisy data, we define the regularization functional based on TV that we can use to clean up the noisy image. For that purpose:
```julia
# Regularization
h = (1f0, 1f0, 1f0)                                                     # Grid spacing
L = 12f0                                                                # Spectral norm of the gradient operator
opt = FISTA_options(L; Nesterov=true,
                       niter=20,
                       reset_counter=10,
                       verbose=false)                                   # FISTA options
g_TV  = gradient_norm(2, 1, n, h; complex=false, gpu=true, options=opt) # TV
```
To keep in mind: the spectral norm of the gradient operator must be known (but that's easy, e.g. ``L=4\sum_i1/h_i^2``). In this example, the input image is real valued, hence `complex=false`. Also, note that we must specify a FISTA solver to use TV. In order to perform TV denoising, type
```julia
# Denoising
λ = 0.5f0*norm(x_clean-x_noisy)^2/g_TV(x_clean) # Denoising weight
x_TV = prox(x_noisy, λ, g_TV)                   # TV denoising
```

We can get an even better result by using a reference volume to guide TV. The ideal reference is the ground-truth! So, for this time, let's cheat by setting:
```julia
# Reference-guided regularization
η = 0.1f0*structural_mean(x_clean)                                                 # Stabilization term
P = structural_weight(x_clean; η=η)                                                # Weight based on a given reference
g_rTV  = gradient_norm(2, 1, n, h; weight=P, complex=false, gpu=true, options=opt) # Reference-guided TV
```
Denoise!
```julia
# Denoising (structure-guided)
λ = 0.5f0*norm(x_clean-x_noisy)^2/g_rTV(x_clean) # Denoising weight
x_rTV = prox(x_noisy, λ, g_rTV)                  # Reference-guided TV denoising
```

In inverse problems, deciding the weight ``\lambda`` of the regularization term ``g`` is no trivial matter. For these reasons, sometime it is preferable to set hard constraints ``g\le\varepsilon``. This package provides the utilities to compute projection operators (as defined in Section [Proximal and projection operators](@ref)), for example:
```julia
# Denoising (structure-guided projection)
ε = 0.5f0*g_rTV(x_clean)             # Noise level
x_rTV_proj = proj(x_noisy, ε, g_rTV) # Projection

# Equivalently: Denoising (structure-guided projection)
C = g_rTV ≤ ε                 # Constraint set
x_rTV_proj = proj(x_noisy, C) # Projection
```

Finally, compare the different results:
```julia
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
```