# Rigid registration utilities

export OptionsRigidRegistration, rigid_registration_options, rigid_registration


# rigid_registration_options(; T::DataType=Float32, niter::Integer=10, verbose::Bool=false, fun_history::Bool=false) = parameter_estimation_options(; niter=niter, steplength=T(1), λ=T(0),
# scaling_diagonal=T(0), scaling_mean=T(0), verbose=verbose, fun_history=fun_history)

# function rigid_registration(u_moving::AbstractArray{CT,3}, u_fixed::AbstractArray{CT,3}, θ::Union{Nothing,AbstractArray{T}}, opt::OptionsParameterEstimation{T}; spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

#     # Initialize variables
#     isnothing(spatial_geometry) ? (X = UtilitiesForMRI.spatial_geometry((T(1),T(1),T(1)), size(u_moving))) : (X = spatial_geometry)
#     kx, ky, kz = k_coord(X; mesh=true)
#     K = cat(reshape(kx, 1, :, 1), reshape(ky, 1, :, 1), reshape(kz, 1, :, 1); dims=3)
#     F = nfft_linop(X, K)
#     d = F*u_fixed

#     # Rigid registration
#     isnothing(θ) && (θ = zeros(T, 1, 6))
#     θ = parameter_estimation(F, u_moving, d, θ, opt)
#     return F'*(F(θ)*u_moving)

# end

mutable struct OptionsRigidRegistration{T<:Real}
    niter::Integer
    η::T
    γ::T
    steplength::T
    verbose::Bool
    fun_history::Union{Nothing,AbstractVector{T}}
end

function rigid_registration_options(; T::DataType=Float32, niter::Integer=10, η::Union{Nothing,Real}=1e-2, γ::Union{Nothing,Real}=0.92, steplength::Real=1.0, verbose::Bool=false, fun_history::Bool=false)
    fun_history ? (fval = Array{typeof(steplength),1}(undef,niter)) : (fval = nothing)
    return OptionsRigidRegistration(niter, T(η), T(γ), T(steplength), verbose, fval)
end

ConvexOptimizationUtils.fun_history(opt::OptionsRigidRegistration) = opt.fun_history

ConvexOptimizationUtils.reset!(opt::OptionsRigidRegistration) = (~isnothing(opt.fun_history) && (opt.fun_history .= 0); return opt)

function rigid_registration(u_moving::AbstractArray{CT,3}, u_fixed::AbstractArray{CT,3}, opt::OptionsRigidRegistration{T}; spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    reset!(opt)
    isnothing(spatial_geometry) ? (X = UtilitiesForMRI.spatial_geometry((T(1),T(1),T(1)), size(u_moving))) : (X = spatial_geometry)
    kx, ky, kz = k_coord(X; mesh=true)
    K = cat(reshape(kx, 1, :, 1), reshape(ky, 1, :, 1), reshape(kz, 1, :, 1); dims=3)
    F = nfft_linop(X, K)
    d = F*u_moving
    θ = zeros(T, 1, 6)
    h = spacing(X)
    η = opt.η*structural_maximum(u_fixed; h=h)
    P = structural_weight(u_fixed; h=h, η=η, γ=opt.γ)*gradient_operator(size(u_moving), h; complex=CT<:Complex)

    # Iterative solution
    @inbounds for n = 1:opt.niter

        # Evaluate forward operator
        Fθu, _, Jθ = ∂(F()*u_moving, θ)

        # Data misfit
        r = P*F'*(Fθu-d)
        (~isnothing(opt.fun_history) || opt.verbose) && (fval_n = T(0.5)*norm(r)^2)

        # Print message
        ~isnothing(opt.fun_history) && (opt.fun_history[n] = fval_n)
        opt.verbose && (@info string("Iter [", n, "/", opt.niter, "], fval = ", fval_n))

        # Compute gradient
        g = Jθ'*(F*(P'*r))

        # Hessian
        H = zeros(T, 6, 6)
        @inbounds for j = 1:6
            ej = zeros(T,1,6); ej[j] = 1
            H[:,j] = vec(Jθ'*(F*(P'*(P*(F'*(Jθ*ej))))))
        end

        # Preconditioning
        g = reshape(H\vec(g), 1, 6)

        # Update
        θ .-= opt.steplength*g

    end

    return F'*(F(θ)*u_moving)

end