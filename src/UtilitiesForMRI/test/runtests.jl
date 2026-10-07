using RetrospectiveMotionCorrectionMRI.UtilitiesForMRI, Test

@testset "UtilitiesForMRI.jl" begin
    @testset "test_spatial_geometry" begin include("./test_spatial_geometry.jl") end
    @testset "test_kspace_geometry" begin include("./test_kspace_geometry.jl") end
    @testset "test_scaling_utils" begin include("./test_scaling_utils.jl") end
    @testset "test_plotting_utils" begin include("./test_plotting_utils.jl") end
    @testset "test_translations" begin include("./test_translations.jl") end
    @testset "test_rotations" begin include("./test_rotations.jl") end
    @testset "test_nfft" begin include("./test_nfft.jl") end
    @testset "test_motion_parameter_utils" begin include("./test_motion_parameter_utils.jl") end
    @testset "test_imagequality_utils" begin include("./test_imagequality_utils.jl") end
end