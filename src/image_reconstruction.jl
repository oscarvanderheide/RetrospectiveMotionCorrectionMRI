# Image reconstruction utilities

export OptionsImageReconstruction
export image_reconstruction_options, image_reconstruction_FISTA_options, image_reconstruction, set_reg_mean


## Reconstruction options

struct AndersonAccelerationOptions{T}
    hist_size::Integer
    β::T
end

abstract type AbstractOptionsImageReconstruction end

mutable struct OptionsImageReconstruction{T}<:AbstractOptionsImageReconstruction
    niter::Integer
    steplength::T
    λ::T
    reg_mean::Union{Nothing,AbstractArray{<:RealOrComplex{T},3}}
    Anderson::Union{Nothing,AndersonAccelerationOptions{T}}
    verbose::Bool
end

function image_reconstruction_options(; niter::Integer=10, steplength::T=1.0, λ::T=0.0, reg_mean::Union{Nothing,AbstractArray}=nothing, hist_size::Union{Nothing,Integer}=nothing, β::Union{Nothing,T}=nothing, verbose::Bool=false) where {T<:Real}
    (~isnothing(hist_size) && ~isnothing(β)) ? (AndersonAcc = AndersonAccelerationOptions(hist_size, β)) : (AndersonAcc = nothing)
    return OptionsImageReconstruction{T}(niter, steplength, λ, reg_mean, AndersonAcc, verbose)
end

set_reg_mean(opt::OptionsImageReconstruction{T}, u::AbstractArray{CT,3}) where {T<:Real,CT<:RealOrComplex{T}} = OptionsImageReconstruction{T}(opt.niter, opt.steplength, opt.λ, u, opt.Anderson, opt.verbose)

mutable struct OptionsImageReconstructionFISTA<:AbstractOptionsImageReconstruction
    niter::Integer
    steplength::Union{Nothing,<:Real}
    niter_spect::Integer
    prox::Function
    Nesterov::Bool
    verbose::Bool
end

image_reconstruction_FISTA_options(; niter::Integer=10, steplength::Union{Nothing,T}=nothing, niter_spect::Integer=3, prox::Function, Nesterov::Bool, verbose::Bool=false) where {T<:Real} = OptionsImageReconstructionFISTA(niter, steplength, niter_spect, prox, Nesterov, verbose)


## Reconstruction algorithms

function image_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, u::AbstractArray{CT,3}, opt::OptionsImageReconstruction) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    isnothing(opt.reg_mean) ? (u_reg_mean = T(0)) : (u_reg_mean = opt.reg_mean)
    fval = Array{T,1}(undef, opt.niter)
    u = deepcopy(u)

    # Initialize Anderson acceleration
    flag_Anderson = ~isnothing(opt.Anderson)
    flag_Anderson && (opt_Anderson = Anderson(; lr=opt.steplength, hist_size=opt.Anderson.hist_size, β=opt.Anderson.β))

    # Iterative solution
    for n = 1:opt.niter

        # Data misfit
        r = F*u-d
        Δu = u.-u_reg_mean
        fval[n] = T(0.5)*norm(r)^2+T(0.5)*opt.λ^2*norm(Δu)^2

        # Print message
        opt.verbose && println("Iter [", n, "/", opt.niter, "], fval = ", fval[n])

        # Gradient
        g = F'*r+opt.λ^2*Δu

        # Update
        flag_Anderson ? update!(opt_Anderson, u, g) : update!(u, opt.steplength*g)

    end

    return u, fval

end

function image_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, u::AbstractArray{CT,3}, opt::OptionsImageReconstructionFISTA) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    fval = Array{T,1}(undef, opt.niter)
    u = deepcopy(u)

    # Estimate steplength if not provided
    if isnothing(opt.steplength)
        opt.verbose && println("Estimating steplength...")
        steplength = T(1)/spectral_radius(F'*F, randn(CT, size(u)); niter=opt.niter_spect)
    else
        steplength = T(opt.steplength)
    end

    # Initialize FISTA updater
    opt_FISTA = FISTA(T(1)/steplength, opt.prox; Nesterov=opt.Nesterov)

    # Iterative solution
    for n = 1:opt.niter

        # Data misfit
        r = F*u-d
        fval[n] = T(0.5)*norm(r)^2

        # Print message
        opt.verbose && println("Iter [", n, "/", opt.niter, "], fval = ", fval[n])

        # Gradient
        g = F'*r

        # Update
        update!(opt_FISTA, u, g)

    end

    ~isnothing(opt.steplength) ? (return (u, fval)) : (return (u, fval, steplength))

end