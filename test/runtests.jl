using FastSolversForWeightedTV, Test

@testset "FastSolversForWeightedTV.jl" begin
    include("./test_gradient.jl")
    include("./test_structural_weight.jl")
    include("./test_tv_norm.jl")
    include("./test_tv_norm_plus_constraints.jl")
end