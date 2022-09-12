# Parameter estimation utilities

export OptionsParameterEstimation, parameter_estimation_options, parameter_estimation, rigid_registration_options, rigid_registration


## Parameter-estimation options

mutable struct HessianRegularizationParameters{T<:Real}
    scaling_diagonal::T
    scaling_mean::T
    scaling_id::T
end

mutable struct OptionsParameterEstimation{T<:Real}
    niter::Integer
    steplength::T
    λ::T
    reg_matrix::Union{Nothing,AbstractMatrix{T}}
    interp_matrix::Union{Nothing,AbstractMatrix{T}}
    reg_Hessian::HessianRegularizationParameters{T}
    verbose::Bool
    fun_history::Union{Nothing,AbstractVector{T}}
end

function parameter_estimation_options(; niter::Integer=10,
                                        steplength::Real=1f0,
                                        λ::Real=0f0,
                                        scaling_diagonal::Real=0f0, scaling_mean::Real=0f0, scaling_id::Real=0f0,
                                        reg_matrix::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                        interp_matrix::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                        verbose::Bool=false,
                                        fun_history::Bool=false)
    fun_history ? (fval = Array{typeof(steplength),1}(undef,niter)) : (fval = nothing)
    return OptionsParameterEstimation(niter, steplength, λ, isnothing(reg_matrix) ? nothing : reg_matrix, isnothing(interp_matrix) ? nothing : interp_matrix, HessianRegularizationParameters(scaling_diagonal, scaling_mean, scaling_id), verbose, fval)
end

ConvexOptimizationUtils.fun_history(opt::OptionsParameterEstimation) = opt.fun_history

ConvexOptimizationUtils.reset!(opt::OptionsParameterEstimation) = (~isnothing(opt.fun_history) && (opt.fun_history .= 0); return opt)


## Parameter-estimation algorithms

function parameter_estimation(F::StructuredNFFTtype2LinOp{T}, u::AbstractArray{CT,3}, d::AbstractArray{CT,2}, initial_estimate::AbstractArray{T,2}, opt::OptionsParameterEstimation{T}) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    reset!(opt)
    θ = deepcopy(initial_estimate)
    interp_flag = ~isnothing(opt.interp_matrix); interp_flag && (Ip = opt.interp_matrix)
    reg_flag    = ~isnothing(opt.reg_matrix) && (opt.λ != T(0)); reg_flag && (D = opt.reg_matrix)

    # Iterative solution
    @inbounds for n = 1:opt.niter

        # Evaluate forward operator
        interp_flag ? (Iθ = reshape(Ip*vec(θ), :, 6)) : (Iθ = θ)
        Fθu, _, Jθ = ∂(F()*u, Iθ)

        # # Calibration ###
        # α = sum(conj(Fθu).*d; dims=2)./sum(conj(Fθu).*Fθu; dims=2) ###
        # A = linear_operator(CT, size(d), size(d), d->α.*d, d->conj(α).*d) ###

        # Data misfit
        # r = A*Fθu-d ###
        r = Fθu-d
        (~isnothing(opt.fun_history) || opt.verbose) && (fval_n = T(0.5)*norm(r)^2)

        # Regularization term
        if reg_flag
            Dθ = reshape(D*vec(θ), :, 6)
            (~isnothing(opt.fun_history) || opt.verbose) && (fval_n += T(0.5)*opt.λ^2*norm(Dθ)^2)
        end

        # Print message
        ~isnothing(opt.fun_history) && (opt.fun_history[n] = fval_n)
        opt.verbose && (@info string("Iter [", n, "/", opt.niter, "], fval = ", fval_n))

        # Compute gradient
        # g = Jθ'*(A'*r) ###
        g = Jθ'*r
        interp_flag && (g = reshape(Ip'*vec(g), :, 6))
        reg_flag && (g .+= opt.λ^2*reshape(D'*vec(Dθ), :, 6))

        # Hessian
        H = sparse_matrix_GaussNewton(Jθ)
        # H = sparse_matrix_GaussNewton(Jθ; W=A)
        interp_flag && (H = Ip'*H*Ip)
        reg_flag && (H .+= opt.λ^2*(D'*D))

        # Marquardt-Levenberg regularization
        if ~isnothing(opt.reg_Hessian)
            H, ΔH = regularize_Hessian!(H; regularization_options=opt.reg_Hessian)
            g .+= reshape(ΔH*vec(θ), :, 6) # consistency correction
        end

        # Preconditioning
        g = reshape(lu(H)\vec(g), :, 6)

        # Update
        θ .-= opt.steplength*g

    end

    return θ

end

function regularize_Hessian!(H::AbstractMatrix{T}; regularization_options::Union{Nothing,HessianRegularizationParameters{T}}=nothing) where {T<:Real}
    isnothing(regularization_options) && (return (H, nothing))
    diagH = reshape(diag(H), :, 6)
    mean_diag = sum(diagH; dims=1)/size(diagH, 1)
    ΔH = spdiagm(vec(regularization_options.scaling_diagonal*diagH.+regularization_options.scaling_mean*mean_diag.+regularization_options.scaling_id))
    H .+= ΔH
    return H, ΔH
end


## Rigid registration

rigid_registration_options(; T::DataType=Float32, niter::Integer=10, verbose::Bool=false, fun_history::Bool=false) = parameter_estimation_options(; niter=niter, steplength=T(1), λ=T(0),
scaling_diagonal=T(0), scaling_mean=T(0), verbose=verbose, fun_history=fun_history)

function rigid_registration(u_moving::AbstractArray{CT,3}, u_fixed::AbstractArray{CT,3}, θ::Union{Nothing,AbstractArray{T}}, opt::OptionsParameterEstimation{T}; spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    isnothing(spatial_geometry) ? (X = UtilitiesForMRI.spatial_geometry((T(1),T(1),T(1)), size(u_moving))) : (X = spatial_geometry)
    kx, ky, kz = k_coord(X; mesh=true)
    K = cat(reshape(kx, 1, :, 1), reshape(ky, 1, :, 1), reshape(kz, 1, :, 1); dims=3)
    F = nfft_linop(X, K)
    d = F*u_fixed

    # Rigid registration
    isnothing(θ) && (θ = zeros(T, 1, 6))
    θ = parameter_estimation(F, u_moving, d, θ, opt)
    return F'*(F(θ)*u_moving)

end