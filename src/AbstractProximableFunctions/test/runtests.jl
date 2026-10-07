using RetrospectiveMotionCorrectionMRI.AbstractProximableFunctions, Test

@testset "AbstractProximableFunctions.jl" begin
    @testset "test_differentiable_functions" begin include("./test_differentiable_functions.jl") end
    @testset "test_projectionable_sets" begin include("./test_projectionable_sets.jl") end
    @testset "test_optimization_utils" begin include("./test_optimization_utils.jl") end
    @testset "test_indicator" begin include("./test_indicator.jl") end
    @testset "test_norms" begin include("./test_norms.jl") end
    @testset "test_mixed_norms" begin include("./test_mixed_norms.jl") end
    @testset "test_weighted_mixed_norms" begin include("./test_weighted_mixed_norms.jl") end
    @testset "test_prox_plus_indicator" begin include("./test_prox_plus_indicator.jl") end
end