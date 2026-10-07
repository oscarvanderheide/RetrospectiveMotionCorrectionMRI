using UtilitiesForMRI, Test

@testset "UtilitiesForMRI.jl" begin
    include("./test_spatial_geometry.jl")
    include("./test_kspace_geometry.jl")
    include("./test_scaling_utils.jl")
    include("./test_plotting_utils.jl")
    include("./test_translations.jl")
    include("./test_rotations.jl")
    include("./test_nfft.jl")
    include("./test_motion_parameter_utils.jl")
    include("./test_plotting_utils.jl")
    include("./test_imagequality_utils.jl")
end