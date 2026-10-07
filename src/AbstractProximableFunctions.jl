module AbstractProximableFunctions

using LinearAlgebra, AbstractLinearOperators, Roots

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./abstract_types.jl")
include("./minimizable_functions.jl")
include("./differentiable_functions.jl")
include("./proximable_functions.jl")
include("./proximable_norms.jl")
include("./projectionable_sets.jl")
include("./misc_utils.jl")

end