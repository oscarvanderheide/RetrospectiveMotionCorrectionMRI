using RetrospectiveMotionCorrectionMRI, Test

# GPU tests require CUDA.jl and NonuniformFFTs.jl, and a functional GPU
const RUN_GPU_TESTS = !isnothing(Base.find_package("CUDA")) && !isnothing(Base.find_package("NonuniformFFTs")) && (@eval(Main, using CUDA); Base.invokelatest(() -> CUDA.functional()))

@testset "RetrospectiveMotionCorrectionMRI.jl" begin
    include("./test_weighted_gradient_operator.jl")
    include("./test_weighted_tv_projection.jl")
    include("./test_leastsquares_misfit.jl")
    include("./test_nfft_jacobian.jl")
    include("./test_gauss_newton.jl")
    include("./test_parameter_estimation_calibration.jl")
    include("./test_toeplitz.jl")
    RUN_GPU_TESTS ? include("./test_gpu.jl") : @info "Skipping GPU tests (no functional CUDA GPU, or CUDA/NonuniformFFTs not installed)"
    include("./test_imagequality_utils.jl")
    include("./test_image_reconstruction.jl")
    include("./test_parameter_estimation.jl")
    include("./test_rigid_registration.jl")
    include("./test_motion_correction.jl")
end