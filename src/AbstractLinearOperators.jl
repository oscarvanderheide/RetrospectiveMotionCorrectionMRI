module AbstractLinearOperators

RealOrComplex{T<:Real} = Union{T,Complex{T}}

# Modules
using LinearAlgebra, SparseArrays, NNlib, cuDNN

# Abstract type
include("./abstract_type.jl")
include("./concrete_type.jl")

# Algebra
include("./linear_algebra.jl")

# Basic examples
include("./linear_operators/basic_linear_operators.jl")
include("./linear_operators/padding_operators.jl")
include("./linear_operators/convolution_operators.jl")
include("./linear_operators/gradient_operator.jl")
include("./linear_operators/Haar_transform.jl")

# Test
include("./test_utils.jl")

end