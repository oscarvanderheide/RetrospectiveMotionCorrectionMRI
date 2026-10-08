using RetrospectiveMotionCorrectionMRI, Test

# GPU tests require CUDA.jl, NonuniformFFTs.jl and cuDNN.jl, and a functional GPU
const RUN_GPU_TESTS = all(!isnothing(Base.find_package(p)) for p in ("CUDA", "NonuniformFFTs", "cuDNN")) && (@eval(Main, using CUDA); Base.invokelatest(() -> CUDA.functional()))
RUN_GPU_TESTS && @eval(Main, using cuDNN, NonuniformFFTs)

@testset "RetrospectiveMotionCorrectionMRI.jl" begin

    # Internal modules (formerly separate packages)
    include("../src/AbstractLinearOperators/test/runtests.jl")
    include("../src/AbstractProximableFunctions/test/runtests.jl")
    include("../src/FastSolversForWeightedTV/test/runtests.jl")
    include("../src/UtilitiesForMRI/test/runtests.jl")

    # RetrospectiveMotionCorrectionMRI
    @testset "test_weighted_gradient_operator" begin include("./test_weighted_gradient_operator.jl") end
    @testset "test_weighted_tv_projection" begin include("./test_weighted_tv_projection.jl") end
    @testset "test_leastsquares_misfit" begin include("./test_leastsquares_misfit.jl") end
    @testset "test_nfft_jacobian" begin include("./test_nfft_jacobian.jl") end
    @testset "test_gauss_newton" begin include("./test_gauss_newton.jl") end
    @testset "test_parameter_estimation_calibration" begin include("./test_parameter_estimation_calibration.jl") end
    @testset "test_toeplitz" begin include("./test_toeplitz.jl") end
    RUN_GPU_TESTS ? (@testset "test_gpu" begin include("./test_gpu.jl") end) : @info "Skipping GPU tests (no functional CUDA GPU, or CUDA/NonuniformFFTs/cuDNN not installed)"
    @testset "test_image_reconstruction" begin include("./test_image_reconstruction.jl") end
    @testset "test_parameter_estimation" begin include("./test_parameter_estimation.jl") end
    @testset "test_rigid_registration" begin include("./test_rigid_registration.jl") end
    @testset "test_motion_correction" begin include("./test_motion_correction.jl") end
end