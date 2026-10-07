using RetrospectiveMotionCorrectionMRI.AbstractLinearOperators, LinearAlgebra, CUDA, Test, Random
CUDA.allowscalar(false)
Random.seed!(42)

@testset "AbstractLinearOperators.jl" begin
    @testset "test_linalg" begin include("./test_linalg.jl") end
    @testset "test_identity" begin include("./test_identity.jl") end
    @testset "test_reshape" begin include("./test_reshape.jl") end
    @testset "test_real2complex" begin include("./test_real2complex.jl") end
    @testset "test_zero_padding" begin include("./test_zero_padding.jl") end
    @testset "test_repeat_padding" begin include("./test_repeat_padding.jl") end
    @testset "test_convolution" begin include("./test_convolution.jl") end
    @testset "test_gradient" begin include("./test_gradient.jl") end
    @testset "test_Haar_transform" begin include("./test_Haar_transform.jl") end
end