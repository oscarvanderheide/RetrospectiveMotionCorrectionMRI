module FastSolversForWeightedTV

using LinearAlgebra, NNlib
using ..AbstractLinearOperators
using ..AbstractProximableFunctions

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./gradient_norm.jl")
include("./gradient_operator.jl")
include("./weighting_operator.jl")
include("./weighted_gradient_operator.jl")
include("./weighted_tv_projection.jl")

end