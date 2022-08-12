using RetrospectiveMotionCorrectionMRI, Test

@testset "RetrospectiveMotionCorrectionMRI.jl" begin
    include("./test_imagequality_utils.jl")
    include("./test_image_reconstruction.jl")
    include("./test_parameter_estimation.jl")
end