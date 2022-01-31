# Parameter estimation utilities

export OptionsParameterEstimation
export parameter_estimation_options, parameter_estimation


## Parameter-estimation options

abstract type AbstractOptionsParameterEstimation{T} end

mutable struct OptionsParameterEstimation{T}<:AbstractOptionsParameterEstimation{T}
    niter::Integer
    steplength::T
    λ::T
    cdiag::T
    cid::T
    reg_matrix::Union{Nothing,AbstractMatrix{T}}
    interp_matrix::Union{Nothing,AbstractMatrix{T}}
    verbose::Bool
end

parameter_estimation_options(; niter::Integer=10, steplength::T=1.0, λ::T=0.0, cdiag::T=0.0, cid::T=0.0, reg_matrix::Union{Nothing,AbstractMatrix{T}}=nothing, interp_matrix::Union{Nothing,AbstractMatrix{T}}=nothing, verbose::Bool=false) where {T<:Real} = OptionsParameterEstimation{T}(niter, steplength, λ, cdiag, cid, reg_matrix, interp_matrix, verbose)


## Parameter-estimation algorithms

function parameter_estimation(F::NFFTParametericLinOp{T}, u::AbstractArray{CT,3}, d::AbstractArray{CT,2}, θ::AbstractArray{T}, opt::AbstractOptionsParameterEstimation{T}) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    fval = Array{T,1}(undef, opt.niter)
    nt, _ = size(F.K)
    interp_flag = ~isnothing(opt.interp_matrix); interp_flag && (Ip = opt.interp_matrix)
    reg_flag = ~isnothing(opt.reg_matrix) && (opt.λ != T(0)); reg_flag && (D = opt.reg_matrix)

    # Iterative solution
    for n = 1:opt.niter

        # Data misfit
        interp_flag ? (θ_ = reshape(Ip*vec(θ), nt, 6)) : (θ_ = θ)
        Fθu, _, Jθ = ∂(F()*u, θ_)
        r = Fθu-d
        fval[n] = T(0.5)*norm(r)^2
        if reg_flag
            Dθ = D*vec(θ_)
            fval[n] += T(0.5)*opt.λ^2*norm(Dθ)^2
        end

        # Print message
        opt.verbose && println("Iter [", n, "/", opt.niter, "], fval = ", fval[n])

        # Gradient
        g = vec(real(Jθ'*r))
        reg_flag && (g .+= opt.λ^2*(D'*Dθ))
        interp_flag && (g = Ip'*g)

        # Hessian preconditioning
        H = reg_Hessian!(sparse_matrix_GaussNewton(Jθ); cdiag=opt.cdiag, cid=opt.cid)
        reg_flag && (H .+= opt.λ^2*(D'*D))
        interp_flag && (H = Ip'*H*Ip)
        g .= lu(H)\g

        # Update
        update!(θ, opt.steplength*g)

    end

    return θ, fval

end

function reg_Hessian!(H::AbstractMatrix{T}; cdiag::T=T(0), cid::T=T(0)) where {T<:Real}
    if cdiag != T(0)
        d = reshape(diag(H), :, 6)
        d_mean = sum(d; dims=1)/size(d,1)
        return H .+= cdiag*spdiagm(vec(d.+cid*d_mean))
    else
        return H
    end
end