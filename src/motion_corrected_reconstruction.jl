# Motion-corrected reconstruction utilities

export OptionsMotionCorrection
export motion_correction_options, motion_corrected_reconstruction


## Motion-corrected reconstruction options

abstract type AbstractOptionsMotionCorrection{T} end

mutable struct OptionsMotionCorrection{T}<:AbstractOptionsMotionCorrection{T}
    niter::Integer
    image_reconstruction::AbstractOptionsImageReconstruction
    parameter_estimation::OptionsParameterEstimation{T}
    verbose::Bool
end

motion_correction_options(; niter::Integer=1, image_reconstruction_options::AbstractOptionsImageReconstruction=image_reconstruction_options(), parameter_estimation_options::OptionsParameterEstimation{T}=parameter_estimation_options(), verbose::Bool=false) where {T<:Real} = OptionsMotionCorrection{T}(niter, image_reconstruction_options, parameter_estimation_options, verbose)


## Motion-corrected reconstruction algorithms

function motion_corrected_reconstruction(F::NFFTParametericLinOp{T}, d::AbstractArray{CT,2}, u::Union{Nothing,AbstractArray{CT,3}}, θ::Union{Nothing,AbstractArray{T}}, opt::OptionsMotionCorrection{T}) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    fval_recon  = Array{Array{T,1},1}(undef, opt.niter)
    fval_parest = Array{Array{T,1},1}(undef, opt.niter-1)
    nt, _ = size(F.K)
    flag_interp = ~isnothing(opt.parameter_estimation.interp_matrix); flag_interp && (Ip = opt.parameter_estimation.interp_matrix)
    isnothing(u) ? (u = zeros(CT, F.X.n)) : (u = deepcopy(u))
    isnothing(θ) ? (flag_interp ? (θ = zeros(T, div(size(Ip,2),6), 6)) : (θ = zeros(T, nt, 6))) : (θ = deepcopy(θ))

    # Iterative solution
    global A = identity_operator(ComplexF32, size(d))
    for n = 1:opt.niter

        # Print message
        opt.verbose && println("* Outer loop: Iter {", n, "/", opt.niter, "}")

        # Image reconstruction
        opt.verbose && println("- Image reconstruction...")
        flag_interp ? (θ_ = reshape(Ip*vec(θ), nt, 6)) : (θ_ = θ)
        u, fval_recon[n], _ = image_reconstruction(F(θ_), d, u, opt.image_reconstruction)

        # Motion-parameter estimation
        (n == opt.niter) && break
        opt.verbose && println("- Motion-parameter estimation...")
        θ, fval_parest[n], A = parameter_estimation(F, u, d, θ, opt.parameter_estimation)

    end

    return u, θ, Dict(:image_reconstruction => fval_recon, :parameter_estimation => fval_parest)

end