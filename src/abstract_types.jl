# Abstract types

abstract type AbstractSpatialGeometry{T<:Real} end
abstract type AbstractCartesianSpatialGeometry{T<:Real}<:AbstractSpatialGeometry{T} end

abstract type AbstractKSpaceSampling{T<:Real} end
abstract type AbstractStructuredKSpaceSampling{T<:Real}<:AbstractKSpaceSampling{T} end

abstract type AbstractNFFTLinOp{T<:Real,XT<:AbstractSpatialGeometry{T},KT<:AbstractKSpaceSampling{T},N,M}<:AbstractLinearOperator{Complex{T},N,M} end