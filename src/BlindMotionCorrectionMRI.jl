module BlindMotionCorrectionMRI

using LinearAlgebra, AbstractLinearOperators, FastSolversForWeightedTV, UtilitiesForMRI, Flux, ImageQualityIndexes, SparseArrays, PyPlot
import Flux.Optimise: update!

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./image_reconstruction.jl")
include("./parameter_estimation.jl")
include("./motion_corrected_reconstruction.jl")
include("./utils.jl")

end