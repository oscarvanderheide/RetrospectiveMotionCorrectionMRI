using JointRetrospectiveMotionCorrectionMRI, Test

@testset "JointRetrospectiveMotionCorrectionMRI.jl" begin
    include("./test_data_residual_loss.jl")
    include("./test_calibration_utils.jl")
    include("./test_optimization_utils.jl")
    include("./test_imagequality_utils.jl")
    include("./test_image_reconstruction.jl")
    include("./test_parameter_estimation.jl")
end