using AbstractLinearOperators, LinearAlgebra, CUDA, Test, Random
CUDA.allowscalar(false)
Random.seed!(42)

@testset "AbstractLinearOperators.jl" begin
    include("./test_linalg.jl")
    include("./test_identity.jl")
    include("./test_reshape.jl")
    include("./test_real2complex.jl")
    include("./test_zero_padding.jl")
    include("./test_repeat_padding.jl")
    include("./test_convolution.jl")
    include("./test_gradient.jl")
    include("./test_Haar_transform.jl")
end