# Image reconstruction utilities

export OptionsImageReconstruction
export image_reconstruction_options, image_reconstruction, set_reg_mean


## Reconstruction options

struct AndersonAccelerationOptions{T}
    hist_size::Integer
    β::T
end

abstract type AbstractOptionsImageReconstruction{T} end

mutable struct OptionsImageReconstruction{T}<:AbstractOptionsImageReconstruction{T}
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


## Reconstruction algorithms

function image_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, u::AbstractArray{CT,3}, opt::OptionsImageReconstruction{T}) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    isnothing(opt.reg_mean) ? (u_reg_mean = T(0)) : (u_reg_mean = opt.reg_mean)
    fval = Array{T,1}(undef, opt.niter)

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
        if opt.verbose
            println("Iter [", n, "/", opt.niter, "], fval = ", fval[n])
        end

        # Gradient
        g = F'*r+opt.λ^2*Δu

        # Update
        flag_Anderson ? update!(opt_Anderson, u, g) : update!(u, opt.steplength*g)

    end

    return u, fval

end