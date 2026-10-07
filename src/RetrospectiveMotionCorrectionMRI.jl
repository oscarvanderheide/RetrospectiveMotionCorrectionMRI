module RetrospectiveMotionCorrectionMRI

using LinearAlgebra, SparseArrays, Reexport

# Include the src code of the submodules that were previously structured as separate packages.
include("AbstractLinearOperators/src/AbstractLinearOperators.jl")
include("UtilitiesForMRI/src/UtilitiesForMRI.jl")
include("AbstractProximableFunctions/src/AbstractProximableFunctions.jl")
include("FastSolversForWeightedTV/src/FastSolversForWeightedTV.jl")

# `using` the submodules to make their functions available in this module.
# Also automatically re-export all the functions from the submodules,
# so that they are available when `using RetrospectiveMotionCorrectionMRI`
@reexport using .AbstractLinearOperators
@reexport using .UtilitiesForMRI
@reexport using .AbstractProximableFunctions
@reexport using .FastSolversForWeightedTV

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

abstract type AbstractImageReconstructionOptions end
abstract type AbstractParameterEstimationOptions end
abstract type AbstractMotionCorrectionOptions end

include("./parameter_estimation.jl")
include("./rigid_registration.jl")
include("./image_reconstruction.jl")
include("./motion_corrected_reconstruction.jl")

end
