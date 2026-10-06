using RetrospectiveMotionCorrectionMRI, Test

@testset "RetrospectiveMotionCorrectionMRI.jl" begin
    include("./test_weighted_gradient_operator.jl")
    include("./test_weighted_tv_projection.jl")
    include("./test_leastsquares_misfit.jl")
    include("./test_nfft_jacobian.jl")
    include("./test_gauss_newton.jl")
    include("./test_imagequality_utils.jl")
    include("./test_image_reconstruction.jl")
    include("./test_parameter_estimation.jl")
    include("./test_rigid_registration.jl")
    include("./test_motion_correction.jl")
end