# Image reconstruction utilities

export image_reconstruction_options, image_reconstruction


mutable struct OptionsImageReconstruction
    prox::AbstractProximableFunction
    options::AbstractArgminOptions
end

function image_reconstruction_options(; prox::AbstractProximableFunction,
                                        Lipschitz_constant::Real,
                                        Nesterov::Bool=true,
                                        reset_counter::Union{Nothing,Integer}=nothing,
                                        niter::Integer,
                                        verbose::Bool=false,
                                        fun_history::Bool=false)
    options = FISTA_options(Lipschitz_constant; Nesterov=Nesterov, reset_counter=reset_counter, niter=niter, verbose=verbose, fun_history=fun_history)
    return OptionsImageReconstruction(prox, options)
end

AbstractProximableFunctions.fun_history(options::OptionsImageReconstruction) = fun_history(options.options)

image_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, initial_estimate::AbstractArray{CT,3}; options::Union{Nothing,OptionsImageReconstruction}=nothing) where {CT<:RealOrComplex} = leastsquares_solve(F, d, options.prox, initial_estimate, options.options)