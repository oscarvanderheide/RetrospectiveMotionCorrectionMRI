using RetrospectiveMotionCorrectionMRI, LinearAlgebra, PythonPlot

using Statistics
using StatsBase

import PythonPlot as pp
using MAT, Statistics

# Run on an NVIDIA GPU? Requires CUDA.jl and NonuniformFFTs.jl in the script environment
# (] add CUDA NonuniformFFTs). Results agree with the CPU up to the NUFFT accuracy.
const use_gpu = false
use_gpu && @eval using CUDA, NonuniformFFTs
to_gpu(x::AbstractArray) = use_gpu ? CUDA.CuArray(x) : x # images and data
to_gpu(F) = use_gpu ? CUDA.cu(F) : F                       # NUFFT operator
to_cpu(x) = Array(x)

function main()
    # CODE:
    # Create a route to the folder where your scans are
    base = "Path_to_your_main_directory"
    
    # Stablish the size of the voxel used in your scan; 1.0f0 = 1m ; 0.001f0 = 1 mm
    voxel_size = 0.001f0   # 1 mm

    # Name of the file to correct
    dataFileName = ("Corrupted.mat")
    path2data = joinpath(base,  dataFileName)

    # Name of the reference
    referenceFilename  = ("reference.mat")
    path2ref  = joinpath(base, referenceFilename)
    
    # Path to the folder with the results
    basepath  = joinpath(base, "corrected/")
    
    # Read the matlab file and it variables
    MotionCorrupted = MAT.matread(path2data)

    ref = MAT.matread(path2ref)

    @info "-------------------------------------"
    @info "STARTING"
    @info "-------------------------------------"

    # Output file name (set below, inside the loop)
    string_out = ""

    # Corrupted and reference images, must match in size
    MtnCorruptedImage = MotionCorrupted["varNameInMatlab"];
    referenceImage = ref["varNameInMatlab"]

    @info "motion size: " size(MtnCorruptedImage) 
    @info "no motion size: " size(referenceImage)
    
    # Obtain each dimmension size
    n = size(MtnCorruptedImage)
    Nx=n[1]
    Ny=n[2]
    Nz=n[3]

    # Establish the FOV, and the origin of the voxel according to you scan
    fov = (Nx*voxel_size, Ny*voxel_size, Nz*voxel_size)
    o = (0f0, 0f0, 0f0)

    X = spatial_geometry(fov, n; origin=o)

    # Y,Z are the phase-encoded dims (where motion shows up between lines)
    phase_encoding = (2,3)
    K = kspace_sampling(X, phase_encoding)

    nt, _ = size(K)
    @info "Number of time instances: $(nt)"

    # NUFFT encoding operator: (image, motion pars per time) -> k-space
    # tol = NUFFT accuracy. 1f-4 is ~3x faster than the library default (1f-6, which is also at the
    # limit of single precision) and changed results by ~3e-5 (relative) in our tests
    F = nfft_linop(X, K; tol=1f-4)

    # Zero motion here, just used to sample k-space from the corrupted image
    θ_true = zeros(Float32, nt, 6)
    ground_truth = referenceImage; 
    
    d = F(θ_true)*MtnCorruptedImage;

    # Move operator, data and reference to the GPU (no-op if use_gpu = false); motion parameters stay on the CPU
    F = to_gpu(F); d = to_gpu(d); ground_truth = to_gpu(ground_truth)

    # L1 (-> ε) controls how much detail/noise is allowed in the image: it sets a
    # threshold to remove most of the noise without erasing important information.
    # L2 (-> η) controls how edge-aware the Total-Variation algorithm is; high values
    # make it less edge-aware, resulting in a blurrier image
    L_vec  = [ 1.0f0]
    L2_vec = [  0.8f0] 

    # Control the number of iterations our loops do
    # LoopIters[1] = N iter image reconstruction
    # LoopIters[2] = N iter motion parameter estimation
    # LoopIters[3] = N iter loop repeats itself
    LoopIters = [3,3,4]

    # Number of movements you estimate the patient did
    nMovements = 128;

    # Optional speed/accuracy settings (both off = previous behavior; both change the result):
    # - warmstart: start each TV-constraint projection from the previous solution. With the default 10
    #   inner iterations the projection is far from converged; warm starting makes it much more accurate
    #   at the same cost (or allows fewer inner iterations, see opt_inner below). Results then depend
    #   on the history of previous calls
    # - toeplitz: evaluate F'F with FFTs instead of NUFFTs during image reconstruction (~15% faster
    #   image reconstruction on CPU, needs 2 extra arrays of 8x the image size in memory)
    warmstart = false
    toeplitz  = false

    for (iter_l, iter_l2) in zip(L_vec, L2_vec)
            
        @info "-------------------------------------"
        @info " Params: L=$iter_l & L2=$iter_l2" 
        @info "-------------------------------------"

        # Nodes where motion is estimated, then interpolated to all nt instants
        ti = Float32.(range(1, nt; length=nMovements))  # Tweek
        t  = Float32.(1:nt)

        # TODO: set your output filename here
        string_out= "YourResultFileNameHere"
        
        # Lipschitz constant for FISTA (unrelated to the L/L2 regularization params above)
        h = spacing(X); 
        L = 4f0*sum(1 ./h.^2) 
        opt_inner = FISTA_options(L; Nesterov=true, niter=10) 

        # L2 (-> η) = how edge-aware the regularization is (high = blurrier)
        η = iter_l2*(1f-2*structural_mean(ground_truth))   
        @info "eta: " * string(η)
        P = structural_weight(ground_truth; η=η) # reference guide

        g = gradient_norm(2,1,size(ground_truth), h; complex=true, weight=P, options=opt_inner, warmstart=warmstart)

        # L1 (-> ε) = how much detail/noise we allow;
        ε = iter_l * g(ground_truth) 
        @info "epsilon: " * string(ε) 
        
        # Constrain reconstruction to g(u) <= ε
        opt_imrecon = image_reconstruction_options(
            prox=indicator(g ≤ ε),
            Nesterov=true,
            niter=LoopIters[1],    # Tweek
            niter_estimate_Lipschitz=3,
            verbose=true,
            fun_history=true,
            toeplitz=toeplitz
        )

        # Interpolates motion pars from nMovements nodes to full nt resolution
        Ip = interpolation1d_motionpars_linop(ti, t)
        # Penalizes abrupt changes in motion pars over time
        D  = derivative1d_motionpars_linop(ti, 2; pars=(true,true,true,true,true,true)) / 4f0

        opt_parest = parameter_estimation_options(
            niter=LoopIters[2],    # Tweek
            steplength=1f0,
            λ=0f0,
            scaling_diagonal=1f-3,
            scaling_mean=1f-4,
            scaling_id=0f0,
            reg_matrix=D,
            interp_matrix=Ip,
            verbose=true,
            fun_history=true
        )

        options = motion_correction_options( # Testing the downscaling and upscaling
            image_reconstruction_options=opt_imrecon,
            parameter_estimation_options=opt_parest,
            niter=LoopIters[3],   #Tweek
            verbose=true,
            fun_history=true
        )

        # Baseline: reconstruction with no motion correction at all
        u_conventional = F' * d

        θ0 = zeros(Float32, length(ti), 6)  # Initial guess for motion parameters (zero motion)
        u0 = to_gpu(zeros(ComplexF32, n))   # Initial image estimate

        u, θ = motion_corrected_reconstruction(F, d, u0, θ0, options)   # Run alternating reconstruction (joint image + motion estimation)
        θ = reshape(Ip*vec(θ), length(t), 6)    # Interpolate estimated motion parameters to full temporal resolution

        # Created dictionary for the name of the matlab variables and store results in string_out file
        outputdata = Dict();
        outputdata["im_motion_parameters"] = θ      # Motion params estimated by the algorithm ( 6 DoF per time instant(t->traslation; r->rotation): tX, tY, tZ, rX, rY, rZ)
        outputdata["corrected_image_after"] = to_cpu(u);    # Corrected image 
        outputdata["corrected_image_before"] = to_cpu(u_conventional);  # Original image
        
        MAT.matwrite(basepath * string_out * ".mat", outputdata)
        
        @info "-------------------------------------------------------------------"
        @info "Finished: L: $iter_l L2: $iter_l2"
        @info "-------------------------------------------------------------------"
    end
end

main()