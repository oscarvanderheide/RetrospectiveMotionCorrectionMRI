# Loss functions

abstract type AbstractLossFunction{T} end
abstract type AbstractLossFunctionMoreau{T}<:AbstractLossFunction{T} end


# Optimizers

abstract type AbstractOptimiserFISTA{T}<:Flux.Optimise.AbstractOptimiser end


# Image reconstruction

abstract type AbstractOptionsImageReconstruction end