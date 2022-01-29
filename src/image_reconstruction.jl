# Image reconstruction utilities

export OptionsReconstructionFISTA, OptionsReconstructionSplitReg, OptionsReconstructionSplitRegAnderson
export FISTA_reconstruction_options, splitreg_reconstruction_options, splitregAnderson_reconstruction_options, FISTA_reconstruction, splitreg_reconstruction


## Reconstruction options

abstract type AbstractOptionsReconstruction{T} end

mutable struct OptionsReconstructionFISTA{T}<:AbstractOptionsReconstruction{T}
    niter::Integer
    L::T
    Nesterov::Bool
    prox::Function # prox(u, λ)  = v
    verbose::Bool
end

FISTA_reconstruction_options(prox::Function; niter::Integer=10, L::T=T(1), Nesterov::Bool=true, verbose::Bool=false) where {T<:Real} = OptionsReconstructionFISTA{T}(niter, L, Nesterov, prox, verbose)

abstract type AbstractOptionsReconstructionSplitReg{T}<:AbstractOptionsReconstruction{T} end

mutable struct OptionsReconstructionSplitReg{T}<:AbstractOptionsReconstructionSplitReg{T}
    niter::Integer
    steplength::T
    λ::T
    u_ref::Union{Nothing,AbstractArray}
    verbose::Bool
end

splitreg_reconstruction_options(; niter::Integer=10, steplength::T=1.0, λ::T=0.0, u_ref::Union{Nothing,AbstractArray}=nothing, verbose::Bool=false) where {T<:Real} = OptionsReconstructionSplitReg{T}(niter, steplength, λ, u_ref, verbose)

mutable struct OptionsReconstructionSplitRegAnderson{T}<:AbstractOptionsReconstructionSplitReg{T}
    opt::OptionsReconstructionSplitReg{T}
    hist_size::Integer
    β::T
end

splitregAnderson_reconstruction_options(; niter::Integer=10, steplength::T=1.0, λ::T=0.0, u_ref::Union{Nothing,AbstractArray}=nothing, hist_size::Integer=5, β::T=1.0, verbose::Bool=false) where {T<:Real} = OptionsReconstructionSplitRegAnderson{T}(OptionsReconstructionSplitReg{T}(niter, steplength, λ, u_ref, verbose), hist_size, β)


## Reconstruction algorithms

function FISTA_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, opt::OptionsReconstructionFISTA{T}; u0::Union{Nothing,AbstractArray{CT,3}}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

    # Setting FISTA solver
    opt_fista = FISTA(opt.L, opt.prox; Nesterov=opt.Nesterov)

    # Initialize variables
    isnothing(u0) ? (u = F'*d) : (u = deepcopy(u0))
    fval = Array{T,1}(undef, opt.niter)
    r = similar(d)
    g = similar(u)

    # Iterative solution
    for n = 1:opt.niter

        # Data misfit
        r .= F*u-d
        fval[n] = T(0.5)*norm(r)^2

        # Print message
        if opt.verbose
            println("Iter [", n, "/", opt.niter, "], fval = ", fval[n])
        end

        # Gradient
        g .= F'*r

        # FISTA update
        update!(opt_fista, u, g)

    end

    return u, fval

end

function splitreg_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, opt::AbstractOptionsReconstructionSplitReg{T}; u0::Union{Nothing,AbstractArray{CT,3}}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    isnothing(u0) ? (u = F'*d) : (u = deepcopy(u0))
    isnothing(opt.u_ref) ? (u_ref = T(0)) : (u_ref = opt.u_ref)
    fval = Array{T,1}(undef, opt.niter)
    r = similar(d)
    g = similar(u)
    Δu = similar(u)

    # Iterative solution
    for n = 1:opt.niter

        # Data misfit
        r .= F*u-d
        Δu .= u.-u_ref
        fval[n] = T(0.5)*norm(r)^2+T(0.5)*opt.λ^2*norm(Δu)^2

        # Print message
        if opt.verbose
            println("Iter [", n, "/", opt.niter, "], fval = ", fval[n])
        end

        # Gradient
        g .= F'*r+opt.λ^2*Δu

        # Update
        u .-= opt.steplength*g

    end

    return u, fval

end

function splitreg_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}, opt::OptionsReconstructionSplitRegAnderson{T}; u0::Union{Nothing,AbstractArray{CT,3}}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    isnothing(u0) ? (u = F'*d) : (u = deepcopy(u0))
    isnothing(opt.opt.u_ref) ? (u_ref = T(0)) : (u_ref = opt.opt.u_ref)
    fval = Array{T,1}(undef, opt.opt.niter)
    r = similar(d)
    g = similar(u)
    Δu = similar(u)

    # Initialize Anderson acceleration
    opt_Anderson = Anderson(; lr=opt.opt.steplength, hist_size=opt.hist_size, β=opt.β)

    # Iterative solution
    for n = 1:opt.opt.niter

        # Data misfit
        r .= F*u-d
        Δu .= u.-u_ref
        fval[n] = T(0.5)*norm(r)^2+T(0.5)*opt.opt.λ^2*norm(Δu)^2

        # Print message
        if opt.opt.verbose
            println("Iter [", n, "/", opt.opt.niter, "], fval = ", fval[n])
        end

        # Gradient
        g .= F'*r+opt.opt.λ^2*Δu

        # Update
        update!(opt_Anderson, u, g)

    end

    return u, fval

end