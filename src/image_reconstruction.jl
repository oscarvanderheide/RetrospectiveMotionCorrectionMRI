# Image reconstruction utilities

export image_reconstruction_options, image_reconstruction


struct ImageReconstructionOptionsFISTA<:AbstractImageReconstructionOptions
    prox::AbstractProximableFunction
    options::ArgminFISTA
end

function image_reconstruction_options(; prox::AbstractProximableFunction,
                               Lipschitz_constant::Union{Nothing,Real}=nothing,
                               Nesterov::Bool=true,
                               reset_counter::Union{Nothing,Integer}=nothing,
                               niter::Union{Nothing,Integer}=nothing,
                               verbose::Bool=false,
                               fun_history::Bool=false)
    return ImageReconstructionOptionsFISTA(prox, FISTA_options(Lipschitz_constant; Nesterov=Nesterov, reset_counter=reset_counter, niter=niter, verbose=verbose, fun_history=fun_history))
end

AbstractProximableFunctions.fun_history(options::ImageReconstructionOptionsFISTA) = fun_history(options.options)

AbstractProximableFunctions.set_Lipschitz_constant(options::ImageReconstructionOptionsFISTA, L::Real) = ImageReconstructionOptionsFISTA(options.prox, set_Lipschitz_constant(options.options, L))

image_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, initial_estimate::AbstractArray{CT,3}, options::ImageReconstructionOptionsFISTA) where {CT<:RealOrComplex} = leastsquares_solve(F, d, options.prox, initial_estimate, options.options)