using RetrospectiveMotionCorrectionMRI, Test

@testset "RetrospectiveMotionCorrectionMRI.jl" begin
    include("./test_imagequality_utils.jl")
    include("./test_image_reconstruction.jl")
    include("./test_parameter_estimation.jl")
    include("./test_rigid_registration.jl")
    include("./test_motion_correction.jl")
end