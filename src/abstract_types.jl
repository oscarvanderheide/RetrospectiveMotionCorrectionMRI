# Loss functions

abstract type AbstractLossFunction{T} end
abstract type AbstractLossFunctionMoreau{T}<:AbstractLossFunction{T} end


# Image reconstruction

abstract type AbstractOptionsImageReconstruction end