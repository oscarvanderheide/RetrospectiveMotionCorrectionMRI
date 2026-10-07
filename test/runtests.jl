using AbstractProximableFunctions, Test

@testset "AbstractProximableFunctions.jl" begin
    include("./test_differentiable_functions.jl")
    include("./test_projectionable_sets.jl")
    include("./test_optimization_utils.jl")
    include("./test_indicator.jl")
    include("./test_norms.jl")
    include("./test_mixed_norms.jl")
    include("./test_weighted_mixed_norms.jl")
    include("./test_prox_plus_indicator.jl")
end