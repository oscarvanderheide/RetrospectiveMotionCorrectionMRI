# Image reconstruction utilities

export OptionsImageReconstructionFISTA
export image_reconstruction_FISTA_options, image_reconstruction


## Reconstruction options

mutable struct OptionsImageReconstructionFISTA{T}<:AbstractOptionsImageReconstruction
    loss::AbstractLossFunction{Complex{T}}
    prox::Function
    niter::Integer
    steplength::Union{Nothing,T}
    niter_estim_Lipschitz_const::Union{Nothing,Integer}
    Nesterov::Bool
    reset_counter::Union{Nothing,Integer}
    calibration::Union{Nothing,AbstractCalibration{T}}
    verbose::Bool
end

image_reconstruction_FISTA_options(loss::AbstractLossFunction{CT},
                                   prox::Function;
                                   niter::Integer=10,
                                   steplength::Union{Nothing,Real}=nothing,
                                   niter_estim_Lipschitz_const::Union{Nothing,Integer}=3,
                                   Nesterov::Bool=true,
                                   reset_counter::Union{Nothing,Integer}=nothing,
                                   calibration::Union{Nothing,AbstractCalibration}=nothing,
                                   verbose::Bool=false) where {T<:Real,CT<:RealOrComplex{T}} =
    OptionsImageReconstructionFISTA{T}(loss, prox, niter, steplength, niter_estim_Lipschitz_const, Nesterov, reset_counter, calibration, verbose)


## Reconstruction algorithms

function image_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, u::AbstractArray{CT,3}, opt::OptionsImageReconstructionFISTA{T}) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    fval = Array{T,1}(undef, opt.niter)
    u = deepcopy(u)

    # Estimate steplength if not provided
    if isnothing(opt.steplength)
        opt.verbose && (@info "Estimating steplength...")
        L = spectral_radius(F'*F, randn(CT, size(u)); niter=opt.niter_estim_Lipschitz_const)*Lipschitz_constant(opt.loss)
    else
        L = T(1)/opt.steplength
    end

    # Initialize FISTA updater
    opt_FISTA = OptimiserFISTA(L, opt.prox; Nesterov=opt.Nesterov, reset_counter=opt.reset_counter)

    # Iterative solution
    isnothing(opt.calibration) && (global A = identity_operator(CT, size(d)))
    for n = 1:opt.niter

        # Evaluate forward operator
        Fu = F*u

        # Calibration (optional)
        ~isnothing(opt.calibration) && (A = opt.calibration(Fu, d))

        # Evaluate objective
        r = A*Fu-d
        fval[n], ∇l = opt.loss(r; Hessian=false)

        # Print message
        opt.verbose && (@info string("Iter [", n, "/", opt.niter, "], fval = ", fval[n]))

        # Compute gradient
        g = F'*A'*∇l

        # Update
        ~isnothing(opt.calibration) && (opt_FISTA.L = L*norm(A)^2)
        update!(opt_FISTA, u, g)

    end

    return u, fval, A

end