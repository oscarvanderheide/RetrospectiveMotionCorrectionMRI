# Parameter estimation utilities

export OptionsParameterEstimation, OptionsSingleUpdateParEstFISTA
export parameter_estimation_options, parameter_estimation, parameter_estimation_proj, rigid_registration_options, rigid_registration
export interp_linear_filling


## Parameter-estimation options

abstract type AbstractOptionsParameterEstimation{T} end

mutable struct OptionsParameterEstimation{T<:Real}<:AbstractOptionsParameterEstimation{T}
    loss::AbstractLossFunction{<:RealOrComplex{T}}
    niter::Integer
    steplength::T
    λ::T
    cdiag::T
    cid::T
    reg_matrix::Union{Nothing,AbstractMatrix{T}}
    interp_matrix::Union{Nothing,AbstractMatrix{T}}
    calibration::Union{Nothing,AbstractCalibration{T}}
    verbose::Bool
end

parameter_estimation_options(T::DataType;
                             loss::AbstractLossFunction=data_residual_loss(Complex{T},2,2),
                             niter::Integer=10,
                             steplength::Real=1.0,
                             λ::Real=0.0,
                             cdiag::Real=0.0, cid::Real=0.0,
                             reg_matrix::Union{Nothing,AbstractMatrix}=nothing,
                             interp_matrix::Union{Nothing,AbstractMatrix}=nothing,
                             calibration::Union{Nothing,AbstractCalibration}=nothing,
                             verbose::Bool=false) =
    OptionsParameterEstimation{T}(loss, niter, T(steplength), T(λ), T(cdiag), T(cid), reg_matrix, interp_matrix, calibration, verbose)


## Parameter-estimation algorithms

function parameter_estimation(F::NFFTParametericLinOp{T}, u::AbstractArray{CT,3}, d::AbstractArray{CT,2}, θ::AbstractArray{T}, opt::AbstractOptionsParameterEstimation{T}) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    θ = deepcopy(θ)
    fval = Array{T,1}(undef, opt.niter)
    nt, _ = size(F.K)
    interp_flag = ~isnothing(opt.interp_matrix); interp_flag && (Ip = opt.interp_matrix)
    reg_flag = ~isnothing(opt.reg_matrix) && (opt.λ != T(0)); reg_flag && (D = opt.reg_matrix)
    loss = opt.loss

    # Iterative solution
    global A
    for n = 1:opt.niter

        # Evaluate forward operator
        interp_flag ? (θ_ = reshape(Ip*vec(θ), nt, 6)) : (θ_ = θ)
        Fθu, _, Jθ = ∂(F()*u, θ_)

        # Calibration (optional)
        ~isnothing(opt.calibration) ? (A = opt.calibration(Fθu, d)) : (A = identity_operator(CT, size(d)))

        # Evaluate objective
        r = A*Fθu-d
        fval[n], ∇l, Hl = loss(r; Hessian=true)
        if reg_flag
            Dθ = D*vec(θ_)
            fval[n] += T(0.5)*opt.λ^2*norm(Dθ)^2
        end

        # Print message
        opt.verbose && println("Iter [", n, "/", opt.niter, "], fval = ", fval[n])

        # Compute gradient
        g = real(Jθ'*A'*∇l)
        reg_flag && (g .+= reshape(opt.λ^2*(D'*Dθ), size(θ_)))

        # Hessian preconditioning
        H, R = reg_Hessian!(sparse_matrix_GaussNewton(Jθ; W=A, H=Hl); cdiag=opt.cdiag, cid=opt.cid)
        ~isnothing(R) && (g .+= reshape(R*vec(θ_), size(g))) # consistency check
        reg_flag && (H .+= opt.λ^2*(D'*D))

        # Interpolation
        if interp_flag
            g = reshape(Ip'*vec(g), size(θ))
            H = Ip'*H*Ip
        end

        # Preconditioning
        g .= reshape(lu(H)\vec(g), size(g))

        # Update
        θ .-= opt.steplength*g

    end

    return θ, fval, A

end

function parameter_estimation_proj(F::NFFTParametericLinOp{T}, u::AbstractArray{CT,3}, d::AbstractArray{CT,2}, θ::AbstractArray{T}, opt::AbstractOptionsParameterEstimation{T}) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    θ = deepcopy(θ)
    fval = Array{T,1}(undef, opt.niter)
    interp_flag = ~isnothing(opt.interp_matrix); interp_flag && (Ip = opt.interp_matrix)

    # Iterative solution
    for n = 1:opt.niter

        # Evaluate forward operator
        Fθu, _, Jθ = ∂(F()*u, θ)

        # Evaluate objective
        r = Fθu-d
        fval[n], ∇l, Hl = opt.loss(r; Hessian=true)
        Δθ = θ-reshape(Ip*(Ip\vec(θ)), size(θ))
        # λ = opt.λ
        λ = opt.λ*sqrt(fval[n]/(T(0.5)*norm(Δθ)^2)); (isinf(λ) || isnan(λ)) && (λ = opt.λ)
        fval[n] += T(0.5)*λ^2*norm(Δθ)^2

        # Print message
        opt.verbose && println("Iter [", n, "/", opt.niter, "], fval = ", fval[n])

        # Compute gradient
        g = real(Jθ'*∇l)+λ^2*Δθ

        # Hessian preconditioning
        H = sparse_matrix_GaussNewton(Jθ; H=Hl)
        H .+= λ^2*sparse(I, size(H))

        # Preconditioning
        g .= reshape(lu(H)\vec(g), size(g))

        # Update
        θ .-= opt.steplength*g

    end

    return θ, fval

end

function reg_Hessian!(H::AbstractMatrix{T}; cdiag::T=T(0), cid::T=T(0)) where {T<:Real}
    if cdiag != T(0)
        d = reshape(diag(H), :, 6)
        d_mean = sum(d; dims=1)/size(d,1)
        D = cdiag*spdiagm(vec(d.+cid*d_mean))
        H .+= D
        return H, D
    else
        return H, nothing
    end
end


## Rigid registration

rigid_registration_options(T::DataType; niter::Integer=10, verbose::Bool=false) = parameter_estimation_options(T; niter=niter, verbose=verbose)

function rigid_registration(u_moving::AbstractArray{CT,3}, u_fixed::AbstractArray{CT,3}, θ::Union{Nothing,AbstractArray{T}}, opt::AbstractOptionsParameterEstimation{T}; h::NTuple{3,T}=(T(1),T(1),T(1))) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    n = size(u_moving)
    X = spatial_sampling(T, n; h=h)
    K = kspace_Cartesian_sampling(X; phase_encoding=(1,3))
    K = UtilitiesForMRI.KSpaceFixedSizeSampling{T}(reshape(K.K, 1, prod(n), 3))
    F = nfft(X, K)
    d = F(zeros(T, 1, 6))*u_fixed

    # Rigid registration
    isnothing(θ) && (θ = zeros(T, 1, 6))
    θ, fval, _ = parameter_estimation(F, u_moving, d, θ, opt)
    u = F(zeros(T, 1, 6))'*(F(θ)*u_moving)

    return u, θ, fval

end


# Parameter processing utilities

function interp_linear_filling(n::NTuple{2,Integer}, θ::AbstractArray{T,2}, fact::Integer; keep_low_freqs::Bool=false) where {T<:Real}
    (fact == 0) && (return θ)

    # Find indexes corresponding to low-frequency region corners
    n1, n2 = n
    k_max = T(pi)
    k1 = range(-k_max, k_max; length=n1)
    k2 = range(-k_max, k_max; length=n2)
    i1 = findfirst(k1 .>= -k_max/2^fact)
    i2 = findlast( k1 .<=  k_max/2^fact)
    j1 = findfirst(k2 .>= -k_max/2^fact)
    j2 = findlast( k2 .<=  k_max/2^fact)

    # Setting parameters to mean value
    θ_ = deepcopy(reshape(θ, n1, n2, 6))
    @inbounds for j = j1:j2, p = 1:6
        θ_[i1:i2, j, p] .= mean(θ_[i1:i2, j, p])
    end
    θ_[:,      1:j1-1, :] .= reshape(θ_[i1, j1, :], 1, 1, 6)
    θ_[1:i1-1, j1,     :] .= reshape(θ_[i1, j1, :], 1, 6)
    @inbounds for j = j1:j2-1, p = 1:6
        t = range(T(0), T(1); length=n2-i2+i1-1)
        θ_[i2+1:end, j,   p] .= vec(θ_[i2, j, p].+t[1:n2-i2].*(θ_[i1, j+1, p]-θ_[i2, j, p]))
        θ_[1:i1-1,   j+1, p] .= vec(θ_[i2, j, p].+t[n2-i2+1:end].*(θ_[i1, j+1, p]-θ_[i2, j, p]))
    end
    θ_[i2+1:end, j2,       :] .= reshape(θ_[i2, j2, :], 1, 6)
    θ_[:,        j2+1:end, :] .= reshape(θ_[i2, j2, :], 1, 1, 6)

    # Restore low frequencies if required
    keep_low_freqs && (θ_[i1:i2, j1:j2, :] .= reshape(θ, n1, n2, 6)[i1:i2, j1:j2, :])

    return reshape(θ_, :, 6)
end