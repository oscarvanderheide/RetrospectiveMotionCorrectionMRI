# [Getting started](@id examples)

In this section, we present a brief step-by-step tutorial on how to perform motion correction for MRI using the tools provided by `RetrospectiveMotionCorrectionMRI`.

Make sure to have installed the package `RetrospectiveMotionCorrectionMRI` and its dependency by following the instructions highlighted [here](@ref install). For plotting, we are also going to make use of `PyPlot`, to install it press `]` in the Julia REPL and type:
```julia
(@v1.8) pkg> add PyPlot
```

Let's start by loading all the needed modules:
```julia
using RetrospectiveMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, AbstractProximableFunctions, LinearAlgebra, PyPlot
```

We can define a simple spatial discretization `X` and ``k``-space trajectory `K` (in this case, corresponding to Cartesian dense sampling). The associated Fourier transform will be `F`:
```julia
# Spatial geometry
fov = (1f0, 2f0, 2f0)
n = (64, 64, 64)
o = (0.5f0, 1f0, 1f0)
X = spatial_geometry(fov, n; origin=o)

# k-space trajectory (Cartesian dense sampling)
phase_encoding = (1, 2)
K = kspace_sampling(X, phase_encoding)
nt, nk = size(K)

# Fourier operator
F = nfft_linop(X, K)
```

We are going now to define a simple 3D image and some rigid motion parameter evolution, which corresponds to a step function in time (e.g. the object moves once halfway through the acquisition). The motion-corrupted data is obtained by perturbing the conventional Fourier transform:
```julia
# Setting image ground-truth (= not motion corrupted)
ground_truth = zeros(ComplexF32, n)
ground_truth[33-10:33+10, 33-10:33+10, 33-10:33+10] .= 1

# Setting simple rigid motion parameters
nt, _ = size(K)
θ_true = zeros(Float32, nt, 6)
θ_true[div(nt,2)+51:end,:] .= reshape([0.0f0, 0.0f0, 0.0f0, Float32(pi)/180*10, 0f0, 0f0], 1, 6)

# Motion-corrupted data
d = F(θ_true)*ground_truth
```

To perform motion correction we need to define several options related to the alternating minimization scheme described in [this methodological section](@ref theory). Refer to that section for in-depth discussion of each of these options:
```julia
# Optimization options:
    ## Image reconstruction options
    h = spacing(X); L = 4f0*sum(1 ./h.^2)
    opt_inner = FISTA_options(L; Nesterov=true, niter=10)
    # η = 1f-2*structural_mean(ground_truth)   # reference guide
    # P = structural_weight(ground_truth; η=η) # reference guide
    P = nothing                                # no reference
    g = gradient_norm(2, 1, size(ground_truth), h; complex=true, weight=P, options=opt_inner)
    ε = 0.8f0*g(ground_truth)
    opt_imrecon = image_reconstruction_options(; prox=indicator(g ≤ ε), Nesterov=true, niter=5, niter_estimate_Lipschitz=3, verbose=true, fun_history=true)

    ## Parameter estimation options
    ti = Float32.(range(1, nt; length=16))
    t = Float32.(1:nt)
    Ip = interpolation1d_motionpars_linop(ti, t)
    D = derivative1d_motionpars_linop(t, 2; pars=(true, true, true, true, true, true))/4f0
    opt_parest = parameter_estimation_options(; niter=5, steplength=1f0, λ=0f0, scaling_diagonal=1f-3, scaling_mean=1f-4, scaling_id=0f0, reg_matrix=D, interp_matrix=Ip, verbose=true, fun_history=true)

    ## Overall options
    options = motion_correction_options(; image_reconstruction_options=opt_imrecon, parameter_estimation_options=opt_parest, niter=40, verbose=true, fun_history=true)
```
Notably, we can opt to perform motion correction with or without a reference guide. This specific behavior is determined by the weight linear operator `P` (comment/uncomment the corresponding line to study the effect on the final results).

The conventional reconstruction is simply performed by:
```julia
# Conventional reconstruction
u_conventional = F'*d
```
The results will be highly corrupted.

To run motion correction:
```julia
# Motion-corrected reconstruction
θ0 = zeros(Float32, length(ti), 6)
u0 = zeros(ComplexF32, n)
u, θ = motion_corrected_reconstruction(F, d, u0, θ0, options)
θ = reshape(Ip*vec(θ), length(t), 6)
```

Finally, we can compare the results by:
```julia
# Plotting
    ## Image
    figure()
    x, y, _ = coord(X)
    extent = (x[1], x[end], y[1], y[end])
    vmin = -0.1; vmax = 1.1
    subplot(1, 3, 1)
    imshow(abs.(u_conventional[:, end:-1:1, 33])'; vmin=vmin, vmax=vmax, extent=extent)
    title("Conventional")
    subplot(1, 3, 2)
    imshow(abs.(u[:, end:-1:1, 33])'; vmin=vmin, vmax=vmax, extent=extent)
    title("Corrected")
    subplot(1, 3, 3)
    imshow(abs.(ground_truth[:, end:-1:1, 33])'; vmin=vmin, vmax=vmax, extent=extent)
    title("Ground-truth")

    ## Motion parameters
    figure()
    plot(θ[:, 4])
    plot(θ_true[:, 4])
```