module AbstractProximableFunctions

using LinearAlgebra, Roots
using ..AbstractLinearOperators

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./abstract_types.jl")
include("./differentiable_functions.jl")
include("./minimizable_functions.jl")
include("./projectionable_sets.jl")
include("./proximable_functions.jl")
include("./proximable_norms.jl")
include("./misc_utils.jl")

end
