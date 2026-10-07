# Load modules
using RetrospectiveMotionCorrectionMRI, LinearAlgebra, PythonPlot

# Spatial geometry
fov = (1f0, 2f0, 2f0)   # Create the FOV in 3D
n = (64, 64, 64)        # Create the 3D space dimensions
o = (0.5f0, 1f0, 1f0)   # Create the origin coordinates
X = spatial_geometry(fov, n; origin=o)


# ----------------------------------------------------

# k-space trajectory (Cartesian dense sampling)
phase_encoding = (1, 2)                 # Indicates which spatial dimensions are phase-encoded (Cartesian sampling)

K = kspace_sampling(X, phase_encoding)  # Generates the Cartesian k-space sampling trajectory associated with X

nt, nk = size(K)

# ----------------------------------------------------
# Fourier operator
# Creates a non-uniform Fourier linear operator F.
# Motion is incorporated later via F(θ).
F = nfft_linop(X, K)

# ----------------------------------------------------

# Setting image ground-truth (= not motion corrupted)
ground_truth = zeros(ComplexF32, n)
ground_truth[33-10:33+10, 33-10:33+10, 33-10:33+10] .= 1

# ----------------------------------------------------

# Setting simple rigid motion parameters
nt, _ = size(K)
θ_true = zeros(Float32, nt, 6)
θ_true[div(nt,2)+51:end,:] .= reshape([0.0f0, 0.0f0, 0.0f0, Float32(pi)/180*10, 0f0, 0f0], 1, 6)

# ----------------------------------------------------

# Motion-corrupted data
# Simulates motion-corrupted k-space data: d = F(θ_true) * ground_truth
d = F(θ_true)*ground_truth

# ---------------------------------------------------# Load modules
using RetrospectiveMotionCorrectionMRI, LinearAlgebra, PythonPlot

# Spatial geometry
fov = (1f0, 2f0, 2f0)   # Create the FOV in 3D
n = (64, 64, 64)        # Create the 3D space dimensions
o = (0.5f0, 1f0, 1f0)   # Create the origin coordinates
X = spatial_geometry(fov, n; origin=o)


# ----------------------------------------------------

# k-space trajectory (Cartesian dense sampling)
phase_encoding = (1, 2)                 # Indicates which spatial dimensions are phase-encoded (Cartesian sampling)

K = kspace_sampling(X, phase_encoding)  # Generates the Cartesian k-space sampling trajectory associated with X

nt, nk = size(K)

# ----------------------------------------------------
# Fourier operator
# Creates a non-uniform Fourier linear operator F.
# Motion is incorporated later via F(θ).
F = nfft_linop(X, K)

# ----------------------------------------------------

# Setting image ground-truth (= not motion corrupted)
ground_truth = zeros(ComplexF32, n)
ground_truth[33-10:33+10, 33-10:33+10, 33-10:33+10] .= 1

# ----------------------------------------------------

# Setting simple rigid motion parameters
nt, _ = size(K)
θ_true = zeros(Float32, nt, 6)
θ_true[div(nt,2)+51:end,:] .= reshape([0f0, 0f0, 0f0, Float32(pi)/180*10, 0f0, 0f0], 1, 6)

# ----------------------------------------------------

# Motion-corrupted data
# Simulates motion-corrupted k-space data: d = F(θ_true) * ground_truth
d = F(θ_true)*ground_truth

# ----------------------------------------------------

# Optimization options:
    ## Image reconstruction options
    h = spacing(X); # Returns the voxel size, in each dimension x, y, z
    L = 4f0*sum(1 ./h.^2) # Estimate Lipschitz constant of the discrete gradient operator (step-size control for FISTA)
    # L too big --> slow convergension
    # L too small --> Diverges

    opt_inner = FISTA_options(L; Nesterov=true, niter=10) # Internal FISTA solver configuration for TV-related subproblems
    # TV means Total Variation, and it measures how much does the image changes voxel-wise. 
    # Noisy images have a big TV; Smooth and edge defined image --> Moderate TV; Completely plain image --> low TV  

    # ------------- Not for this example -----------------------
    # η = 1f-2*structural_mean(ground_truth)   # reference guide
    # P = structural_weight(ground_truth; η=η) # reference guide

    # No structural weighting (standard TV regularization)
    P = nothing                                # no reference

    # Define isotropic Total Variation functional (TV regularizer)
    g = gradient_norm(2, 1, size(ground_truth), h; complex=true, weight=P, options=opt_inner) # Preferences for the image after the calcs

    # Set TV constraint radius (80% of ground-truth TV)
    ε = 0.8f0*g(ground_truth) # Max detail

    # prox=indicator(g ≤ ε) --> rule for the images to adjust. Quality filter 
    opt_imrecon = image_reconstruction_options(; prox=indicator(g ≤ ε), Nesterov=true, niter=1, niter_estimate_Lipschitz=3, verbose=true, fun_history=true)
 
    ## Parameter estimation options

    # Reduces the problem dimensions by only taking "length" points out of the K-space
    ti = Float32.(range(1, nt; length=16))  # Define length temporal control points for motion parameter estimation

    t = Float32.(1:nt)  # Full temporal grid

    Ip = interpolation1d_motionpars_linop(ti, t) # Linear interpolation operator from control points to full time grid

    D = derivative1d_motionpars_linop(t, 2; pars=(true, true, true, true, true, true))/4f0 # Second-order temporal derivative operator (for motion smoothness regularization)

    # Configure motion parameter optimization (gradient-based updates)
    # niter --> Nº steps
    # steplength --> step size
    # λ=0f0 -->
    # reg_matrix=D -->
    opt_parest = parameter_estimation_options(; niter=5, steplength=1f0, λ=0f0, scaling_diagonal=1f-3, scaling_mean=1f-4, scaling_id=0f0, reg_matrix=D, interp_matrix=Ip, verbose=true, fun_history=true)

    ## Overall options
    options = motion_correction_options(; image_reconstruction_options=opt_imrecon, parameter_estimation_options=opt_parest, niter=10, verbose=true, fun_history=true)

# ----------------------------------------------------

# Conventional reconstruction (no motion correction)
u_conventional = F'*d

# ----------------------------------------------------

# Motion-corrected reconstruction
θ0 = zeros(Float32, length(ti), 6)  # Initial guess for motion parameters (zero motion)
u0 = zeros(ComplexF32, n)           # Initial image estimate
u, θ = motion_corrected_reconstruction(F, d, u0, θ0, options)   # Run alternating reconstruction (joint image + motion estimation)
θ = reshape(Ip*vec(θ), length(t), 6)    # Interpolate estimated motion parameters to full temporal resolution

# ----------------------------------------------------

# Plotting
    # Image
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

    # Motion parameters
    figure()
    plot(θ[:, 4], label="Estimated")
    plot(θ_true[:, 4], label="Real")
    legend()
    xlabel("Time")
    ylabel("Rotation (rad)")
    title("Movement stimation")-

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

# ----------------------------------------------------

# Conventional reconstruction
u_conventional = F'*d

# ----------------------------------------------------

# Motion-corrected reconstruction
θ0 = zeros(Float32, length(ti), 6)
u0 = zeros(ComplexF32, n)
u, θ = motion_corrected_reconstruction(F, d, u0, θ0, options)
θ = reshape(Ip*vec(θ), length(t), 6)

# ----------------------------------------------------

# Plotting
    # Image
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

    # Motion parameters
    figure()
    plot(θ[:, 4], label="Estimated")
    plot(θ_true[:, 4], label="Real")
    legend()
    xlabel("Time")
    ylabel("Rotation (rad)")
    title("Movement stimation")