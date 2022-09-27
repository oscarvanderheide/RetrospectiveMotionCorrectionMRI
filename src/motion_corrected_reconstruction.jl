# Motion-corrected reconstruction utilities

export OptionsMotionCorrection, OptionsMotionCorrectionADMM, motion_correction_options, motion_correction_options_ADMM, motion_corrected_reconstruction


## Motion-corrected reconstruction options

mutable struct OptionsMotionCorrection{T<:Real}
    opt_imrecon::OptionsImageReconstruction{T}
    opt_parest::OptionsParameterEstimation{T}
    niter::Integer
    niter_estimate_Lipschitz::Union{Nothing,Integer}
    verbose::Bool
    fun_history::Union{Nothing,AbstractArray}
end

motion_correction_options(; image_reconstruction_options::OptionsImageReconstruction{T}, parameter_estimation_options::OptionsParameterEstimation{T}, niter::Integer, niter_estimate_Lipschitz::Union{Nothing,Integer}=nothing, verbose::Bool=false, fun_history::Bool=false) where {T<:Real} = OptionsMotionCorrection{T}(image_reconstruction_options, parameter_estimation_options, niter, niter_estimate_Lipschitz, verbose, fun_history ? Vector{NTuple{2,Any}}(undef,niter) : nothing)

ConvexOptimizationUtils.reset!(opt::OptionsMotionCorrection{T}) where {T<:Real} = (opt.fun_history = Vector{NTuple{2,Any}}(undef,opt.niter); return opt)

ConvexOptimizationUtils.fun_history(opt::OptionsMotionCorrection{T}) where {T<:Real} = opt.fun_history

mutable struct OptionsMotionCorrectionADMM{T<:Real}
    opt_imrecon::OptionsImageReconstruction{T}
    opt_parest::OptionsParameterEstimation{T}
    niter::Integer
    niter_estimate_Lipschitz::Union{Nothing,Integer}
    ρ::T
    verbose::Bool
    fun_history::Union{Nothing,AbstractArray}
end

motion_correction_options_ADMM(; image_reconstruction_options::OptionsImageReconstruction{T}, parameter_estimation_options::OptionsParameterEstimation{T}, niter::Integer, niter_estimate_Lipschitz::Union{Nothing,Integer}=nothing, ρ::T=T(0), verbose::Bool=false, fun_history::Bool=false) where {T<:Real} = OptionsMotionCorrectionADMM{T}(image_reconstruction_options, parameter_estimation_options, niter, niter_estimate_Lipschitz, ρ, verbose, fun_history ? Vector{NTuple{2,Any}}(undef,niter) : nothing)

ConvexOptimizationUtils.reset!(opt::OptionsMotionCorrectionADMM{T}) where {T<:Real} = (opt.fun_history = Vector{NTuple{2,Any}}(undef,opt.niter); return opt)

ConvexOptimizationUtils.fun_history(opt::OptionsMotionCorrectionADMM{T}) where {T<:Real} = opt.fun_history


## Motion-corrected reconstruction algorithms

function motion_corrected_reconstruction(F::StructuredNFFTtype2LinOp{T}, d::AbstractArray{CT,2}, u::AbstractArray{CT,3}, θ::AbstractArray{T,2}, opt::OptionsMotionCorrection{T}) where {T<:Real,CT<:RealOrComplex{T}}

    reset!(opt)
    flag_interp = ~isnothing(opt.opt_parest.interp_matrix)
    flag_interp && (Ip = opt.opt_parest.interp_matrix)

    # Iterative solution
    for n = 1:opt.niter

        # Print message
        opt.verbose && (@info string("*** Outer loop: Iter {", n, "/", opt.niter, "}"))

        # Image reconstruction
        opt.verbose && (@info string("--- Image reconstruction..."))
        flag_interp ? (θ_ = reshape(Ip*vec(θ), :, 6)) : (θ_ = θ); Fθ = F(θ_)
        # Fθu = Fθ*u ###
        # (norm(u) != CT(0)) ? (α = sum(conj(Fθu).*d; dims=2)./sum(conj(Fθu).*Fθu; dims=2)) : (α = 1) ###
        # A = linear_operator(CT, size(d), size(d), d->α.*d, d->conj(α).*d) ###
        ~isnothing(opt.niter_estimate_Lipschitz) && (opt.opt_imrecon.opt.Lipschitz_constant = T(1.1)*spectral_radius(Fθ*Fθ'; niter=opt.niter_estimate_Lipschitz))
        # AFθ = A*Fθ ###
        # ~isnothing(opt.niter_estimate_Lipschitz) && (opt.opt_imrecon.opt.Lipschitz_constant = T(1.1)*spectral_radius(AFθ*AFθ'; niter=opt.niter_estimate_Lipschitz)) ###
        u = image_reconstruction(Fθ, d, u, opt.opt_imrecon)
        # u = image_reconstruction(AFθ, d, u, opt.opt_imrecon) ###

        # Motion-parameter estimation
        (n == opt.niter) && break
        opt.verbose && (@info string("--- Motion-parameter estimation..."))
        θ = parameter_estimation(F, u, d, θ, opt.opt_parest)

        # Saving history
        ~isnothing(opt.fun_history) && (opt.fun_history[n] = (deepcopy(fun_history(opt.opt_imrecon)), deepcopy(fun_history(opt.opt_parest))))

    end

    return u, θ

end

function motion_corrected_reconstruction(F::StructuredNFFTtype2LinOp{T}, d::AbstractArray{CT,2}, u::AbstractArray{CT,3}, θ::AbstractArray{T,2}, opt::OptionsMotionCorrectionADMM{T}) where {T<:Real,CT<:RealOrComplex{T}}

    reset!(opt)
    flag_interp = ~isnothing(opt.opt_parest.interp_matrix)
    flag_interp && (Ip = opt.opt_parest.interp_matrix)
    y = similar(d); fill!(y, 0) # Lagrange multiplier
    g = deepcopy(opt.opt_imrecon.prox)
    ρ = opt.ρ

    # Iterative solution
    for n = 1:opt.niter

        # Print message
        opt.verbose && (@info string("*** Outer loop: Iter {", n, "/", opt.niter, "}"))

        # Image reconstruction
        opt.verbose && (@info string("--- Image reconstruction..."))
        flag_interp ? (θ_ = reshape(Ip*vec(θ), :, 6)) : (θ_ = θ); Fθ = F(θ_)
        ~isnothing(opt.niter_estimate_Lipschitz) && (opt.opt_imrecon.opt.Lipschitz_constant = T(1.1)*spectral_radius(Fθ*Fθ'; niter=opt.niter_estimate_Lipschitz))
        opt.opt_imrecon.prox = 1/ρ*g
        u = image_reconstruction(Fθ, d-y, u, opt.opt_imrecon)

        # Motion-parameter estimation
        (n == opt.niter) && break
        opt.verbose && (@info string("--- Motion-parameter estimation..."))
        θ = parameter_estimation(F, u, d-y, θ, opt.opt_parest)

        # Multiplier update
        flag_interp ? (θ_ = reshape(Ip*vec(θ), :, 6)) : (θ_ = θ); Fθ = F(θ_)
        y .+= ρ*(Fθ*u-d)
        # y .+= Fθ*u-d

        # Saving history
        ~isnothing(opt.fun_history) && (opt.fun_history[n] = (deepcopy(fun_history(opt.opt_imrecon)), deepcopy(fun_history(opt.opt_parest))))

    end

    return u, θ

end