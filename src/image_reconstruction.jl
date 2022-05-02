# Image reconstruction utilities

export OptionsImageReconstruction
export image_reconstruction_options, image_reconstruction_FISTA_options, image_reconstruction


## Reconstruction options

abstract type AbstractOptionsImageReconstruction end

mutable struct OptionsImageReconstructionFISTA{T}<:AbstractOptionsImageReconstruction
    loss::AbstractLossFunction{Complex{T}}
    niter::Integer
    steplength::Union{Nothing,T}
    niter_EstLipschitzConst::Integer
    prox::Function
    Nesterov::Bool
    calibration::Union{Nothing,AbstractCalibration{T}}
    verbose::Bool
end

image_reconstruction_FISTA_options(T::DataType;
                                   loss::AbstractLossFunction=MixedNorm{<:Complex,2,2},
                                   niter::Integer=10,
                                   steplength::Union{Nothing,Real}=nothing,
                                   niter_EstLipschitzConst::Integer=3,
                                   prox::Function,
                                   Nesterov::Bool=true,
                                   calibration::Union{Nothing,AbstractCalibration}=nothing,
                                   verbose::Bool=false) =
    OptionsImageReconstructionFISTA{T}(loss, niter, isnothing(steplength) ? nothing : T(steplength), niter_EstLipschitzConst, prox, Nesterov, calibration, verbose)


## Reconstruction algorithms

function image_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, u::AbstractArray{CT,3}, opt::OptionsImageReconstructionFISTA{T}) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    fval = Array{T,1}(undef, opt.niter)
    u = deepcopy(u)
    loss = opt.loss

    # Estimate steplength if not provided
    if isnothing(opt.steplength)
        opt.verbose && println("Estimating steplength...")
        LipConst = spectral_radius(F'*F, randn(CT, size(u)); niter=opt.niter_EstLipschitzConst)*Lipschitz_constant(loss)
    else
        LipConst = T(1)/opt.steplength
    end

    # Initialize FISTA updater
    opt_FISTA = FISTA(LipConst, opt.prox; Nesterov=opt.Nesterov)

    # Iterative solution
    global A
    isnothing(opt.calibration) && (A = identity_operator(CT, size(d)))
    for n = 1:opt.niter

        # Evaluate forward operator
        Fu = F*u

        # Calibration (optional)
        ~isnothing(opt.calibration) && (A = opt.calibration(Fu, d))

        # Evaluate objective
        r = A*Fu-d
        fval[n], ∇l = loss(r; Hessian=false)

        # Print message
        opt.verbose && println("Iter [", n, "/", opt.niter, "], fval = ", fval[n])

        # Compute gradient
        g = F'*A'*∇l

        # Update
        ~isnothing(opt.calibration) && (opt_FISTA.L = LipConst*norm(A)^2)
        update!(opt_FISTA, u, g)

    end

    return u, fval, A

end