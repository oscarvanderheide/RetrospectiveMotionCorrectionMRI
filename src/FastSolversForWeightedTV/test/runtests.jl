using RetrospectiveMotionCorrectionMRI.FastSolversForWeightedTV, Test

@testset "FastSolversForWeightedTV.jl" begin
    @testset "test_gradient" begin include("./test_gradient.jl") end
    @testset "test_structural_weight" begin include("./test_structural_weight.jl") end
    @testset "test_tv_norm" begin include("./test_tv_norm.jl") end
    @testset "test_tv_norm_plus_constraints" begin include("./test_tv_norm_plus_constraints.jl") end
end