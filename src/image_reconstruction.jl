# Image reconstruction utilities

export OptionsReconstructionFISTA
export FISTA_reconstruction_options, FISTA_reconstruction


abstract type AbstractOptionsReconstruction{T} end

mutable struct OptionsReconstructionFISTA{T}<:AbstractOptionsReconstruction{T}
    niter::Integer
    L::T
    Nesterov::Bool
    prox::Function # prox(u, λ)  = v
    verbose::Bool
end

FISTA_reconstruction_options(prox::Function; niter::Integer=10, L::T=T(1), Nesterov::Bool=true, verbose::Bool=false) where {T<:Real} = OptionsReconstructionFISTA{T}(niter, L, Nesterov, prox, verbose)

function FISTA_reconstruction(F::AbstractLinearOperator{CT,3,2}, d::AbstractArray{CT,2}; u0::Union{Nothing,AbstractArray{CT,3}}=nothing, opt::OptionsReconstructionFISTA{T}=reconstruction_options((u,λ)->u)) where {T<:Real,CT<:RealOrComplex{T}}

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