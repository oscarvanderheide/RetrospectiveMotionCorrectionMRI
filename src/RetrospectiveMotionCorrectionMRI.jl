module RetrospectiveMotionCorrectionMRI

using LinearAlgebra, AbstractLinearOperators, FastSolversForWeightedTV, AbstractProximableFunctions, UtilitiesForMRI, SparseArrays, ImageQualityIndexes

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

abstract type AbstractImageReconstructionOptions end
abstract type AbstractParameterEstimationOptions end
abstract type AbstractMotionCorrectionOptions end

include("./imagequality_utils.jl")
include("./parameter_estimation.jl")
include("./rigid_registration.jl")
include("./image_reconstruction.jl")
include("./motion_corrected_reconstruction.jl")

end