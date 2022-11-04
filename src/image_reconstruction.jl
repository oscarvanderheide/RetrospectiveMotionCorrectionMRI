# Image reconstruction utilities

export image_reconstruction_options, image_reconstruction


mutable struct OptionsImageReconstruction{T<:Real}
    prox::AbstractProximableFunction{CT,N} where {CT<:RealOrComplex{T},N}
    optimizer::OptimizerFISTA{T}
end

function image_reconstruction_options(; prox::AbstractProximableFunction,
                                        Lipschitz_constant::Real,
                                        Nesterov::Bool=true,
                                        reset_counter::Union{Nothing,Integer}=nothing,
                                        niter::Integer,
                                        verbose::Bool=false,
                                        fun_history::Bool=false)
    optimizer = FISTA_optimizer(Lipschitz_constant; Nesterov=Nesterov, reset_counter=reset_counter, niter=niter, verbose=verbose, fun_history=fun_history)
    return OptionsImageReconstruction(prox, optimizer)
end

ConvexOptimizationUtils.reset!(options::OptionsImageReconstruction) = (reset!(options.optimizer); return options)

ConvexOptimizationUtils.fun_history(options::OptionsImageReconstruction) = fun_history(options.optimizer)

image_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, initial_estimate::AbstractArray{CT,3}; options::Union{Nothing,OptionsImageReconstruction{T}}=nothing) where {T<:Real,CT<:RealOrComplex{T}} = leastsquares_solve(F, d, options.prox, initial_estimate, reset!(options.optimizer))