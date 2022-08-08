using JointRetrospectiveMotionCorrectionMRI, Test

@testset "JointRetrospectiveMotionCorrectionMRI.jl" begin
    include("./test_spatial_geometry.jl")
    include("./test_kspace_geometry.jl")
    # include("./test_motionpars_utils.jl")
    # include("./test_nfft.jl")
    # include("./test_rotations.jl")
    include("./test_translations.jl")
    include("./test_scaling_utils.jl")
    include("./test_plotting_utils.jl")
end