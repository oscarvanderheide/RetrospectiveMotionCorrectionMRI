# Image reconstruction utilities

export image_reconstruction_options, image_reconstruction


mutable struct OptionsImageReconstruction{T<:Real}
    opt::OptimizerFISTA{T,Nothing}
    prox::ProximableFunction{CT,N} where {CT<:RealOrComplex{T},N}
end

function image_reconstruction_options(; prox::ProximableFunction,
                                        Lipschitz_constant::Real,
                                        Nesterov::Bool=true,
                                        reset_counter::Union{Nothing,Integer}=nothing,
                                        niter::Integer,
                                        verbose::Bool=false,
                                        fun_history::Bool=false)
    opt = FISTA_optimizer(Lipschitz_constant; Nesterov=Nesterov, reset_counter=reset_counter, niter=niter, verbose=verbose, fun_history=fun_history)
    return OptionsImageReconstruction(opt, prox)
end

ConvexOptimizationUtils.reset!(opt::OptionsImageReconstruction) = (reset!(opt.opt); return opt)

ConvexOptimizationUtils.fun_history(opt::OptionsImageReconstruction) = fun_history(opt.opt)

image_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, initial_estimate::AbstractArray{CT,3}, opt::OptionsImageReconstruction{T}) where {T<:Real,CT<:RealOrComplex{T}} = leastsquares_solve(F, d, initial_estimate, reset!(opt.opt); prox=opt.prox)