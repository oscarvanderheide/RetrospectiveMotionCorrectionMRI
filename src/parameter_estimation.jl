# Parameter estimation utilities

export OptionsParameterEstimation, parameter_estimation_options, parameter_estimation, rigid_registration_options, rigid_registration


## Parameter-estimation options

mutable struct OptionsParameterEstimation{T<:Real}
    niter::Integer
    steplength::T
    λ::T
    cdiag::T
    cid::Union{T,NTuple{6,T}}
    reg_matrix::Union{Nothing,AbstractMatrix{T}}
    interp_matrix::Union{Nothing,AbstractMatrix{T}}
    verbose::Bool
    fun_history::Union{Nothing,AbstractVector{T}}
end

function parameter_estimation_options(; niter::Integer=10,
                                        steplength::Real=1f0,
                                        λ::Real=0f0,
                                        cdiag::Real=0f0, cid::Union{Real,NTuple{6,<:Real}}=0f0,
                                        reg_matrix::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                        interp_matrix::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                        verbose::Bool=false,
                                        fun_history::Bool=false)
    fun_history ? (fval = Array{typeof(steplength),1}(undef,niter)) : (fval = nothing)
    return OptionsParameterEstimation(niter, steplength, λ, cdiag, cid, isnothing(reg_matrix) ? nothing : reg_matrix, isnothing(interp_matrix) ? nothing : interp_matrix, verbose, fval)
end

ConvexOptimizationUtils.fun_history(opt::OptionsParameterEstimation) = opt.fun_history

ConvexOptimizationUtils.reset!(opt::OptionsParameterEstimation) = (~isnothing(opt.fun_history) && (opt.fun_history .= 0); return opt)


## Parameter-estimation algorithms

function parameter_estimation(F::StructuredNFFTtype2LinOp{T}, u::AbstractArray{CT,3}, d::AbstractArray{CT,2}, initial_estimate::AbstractArray{T}, opt::OptionsParameterEstimation{T}) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    reset!(opt)
    θ = deepcopy(initial_estimate)
    nt = size(F.kcoord, 1)
    interp_flag = ~isnothing(opt.interp_matrix); interp_flag && (Ip = opt.interp_matrix)
    reg_flag = ~isnothing(opt.reg_matrix) && (opt.λ != T(0)); reg_flag && (D = opt.reg_matrix)
    gθ = similar(initial_estimate, nt, 6)
    interp_flag ? (g = similar(initial_estimate)) : (g = gθ)
    interp_flag ? (Iθ = similar(initial_estimate, nt, 6)) : (Iθ = θ)
    r = similar(d)

    # Iterative solution
    @inbounds for n = 1:opt.niter

        # Evaluate forward operator
        interp_flag && (Iθ .= reshape(Ip*vec(θ), nt, 6))
        Fθu, _, Jθ = ∂(F()*u, Iθ)

        # # Calibration ###
        # α = sum(conj(Fθu).*d; dims=2)./sum(conj(Fθu).*Fθu; dims=2) ###
        # A = linear_operator(CT, size(d), size(d), d->α.*d, d->conj(α).*d) ###

        # Data misfit
        # r .= A*Fθu-d ###
        r .= Fθu-d
        (~isnothing(opt.fun_history) || opt.verbose) && (fval_n = T(0.5)*norm(r)^2)

        # Regularization term
        if reg_flag
            Dθ = D*vec(Iθ)
            (~isnothing(opt.fun_history) || opt.verbose) && (fval_n += T(0.5)*opt.λ^2*norm(Dθ)^2)
        end

        # Print message
        ~isnothing(opt.fun_history) && (opt.fun_history[n] = fval_n)
        opt.verbose && (@info string("Iter [", n, "/", opt.niter, "], fval = ", fval_n))

        # Compute gradient
        # gθ .= Jθ'*(A'*r) ###
        gθ .= Jθ'*r
        reg_flag && (gθ .+= reshape(opt.λ^2*(D'*Dθ), nt, 6))

        # Hessian
        H = sparse_matrix_GaussNewton(Jθ)
        # H = sparse_matrix_GaussNewton(Jθ; W=A)
        reg_flag && (H .+= opt.λ^2*(D'*D))

        # Hessian preconditioning
        # H, R = regularize_Hessian!(sparse_matrix_GaussNewton(Jθ; W=A); c_diag=opt.cdiag, c_id=opt.cid) ###
        H, R = regularize_Hessian!(sparse_matrix_GaussNewton(Jθ); c_diag=opt.cdiag, c_id=opt.cid)
        ~isnothing(R) && (gθ .+= reshape(R*vec(Iθ), nt, 6)) # consistency correction

        # Interpolation
        if interp_flag
            g .= Ip'*vec(gθ)
            H = Ip'*H*Ip
        end

        # Preconditioning
        g .= reshape(lu(H)\vec(g), size(g))

        # Update
        θ .-= opt.steplength*g

    end

    return θ

end

# function regularize_Hessian!(H::AbstractMatrix{T}; cdiag::T=T(0), cid::T=T(0)) where {T<:Real}
#     if cdiag != T(0)
#         d = reshape(diag(H), :, 6)
#         d_mean = sum(d; dims=1)/size(d,1)
#         D = cdiag*spdiagm(vec(d.+cid*d_mean))
#         H .+= D
#         return H, D
#     else
#         return H, nothing
#     end
# end

function regularize_Hessian!(H::AbstractMatrix{T}; c_diag::T=T(0), c_id::Union{T,NTuple{6,T}}=T(0)) where {T<:Real}
    d = reshape(diag(H), :, 6)
    c_id isa T && (c_id = (c_id, c_id, c_id, c_id, c_id, c_id))
    c_id = reshape([c_id...], 1, 6)
    D = spdiagm(vec(c_diag*d.+c_id))
    H .+= D
    return H, D
end


## Rigid registration

rigid_registration_options(; T::DataType=Float32, niter::Integer=10, verbose::Bool=false, fun_history::Bool=false) = parameter_estimation_options(; niter=niter, steplength=T(1), λ=T(0),
cdiag=T(0), cid=T(0), verbose=verbose, fun_history=fun_history)

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