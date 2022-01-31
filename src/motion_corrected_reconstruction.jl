# Motion-corrected reconstruction utilities

export OptionsMotionCorrection
export motion_correction_options, motion_corrected_reconstruction


## Motion-corrected reconstruction options

abstract type AbstractOptionsMotionCorrection{T} end

mutable struct OptionsMotionCorrection{T}<:AbstractOptionsMotionCorrection{T}
    niter::Integer
    denoiser::Function
    image_reconstruction::OptionsImageReconstruction{T}
    parameter_estimation::OptionsParameterEstimation{T}
    verbose::Bool
end

motion_correction_options(; niter::Integer=1, denoiser::Function=identity, image_reconstruction_options::OptionsImageReconstruction{T}=image_reconstruction_options(), parameter_estimation_options::OptionsParameterEstimation{T}=parameter_estimation_options(), verbose::Bool=false) where {T<:Real} = OptionsMotionCorrection{T}(niter, denoiser, image_reconstruction_options, parameter_estimation_options, verbose)


## Motion-corrected reconstruction algorithms

function motion_corrected_reconstruction(F::NFFTParametericLinOp{T}, d::AbstractArray{CT,2}, u::AbstractArray{CT,3}, θ::AbstractArray{T}, opt::OptionsMotionCorrection{T}) where {T<:Real,CT<:RealOrComplex{T}}

    # Initialize variables
    fval_recon  = Array{Array{T,1},1}(undef, opt.niter)
    fval_parest = Array{Array{T,1},1}(undef, opt.niter-1)
    nt, _ = size(F.K)
    flag_interp = ~isnothing(opt.parameter_estimation.interp_matrix); flag_interp && (Ip = opt.parameter_estimation.interp_matrix)

    # Iterative solution
    for n = 1:opt.niter

        # Print message
        opt.verbose && println("* Outer loop: Iter {", n, "/", opt.niter, "}")

        # Denoising
        opt.verbose && println("- Denoising...")
        u0 = opt.denoiser(u)

        # Image reconstruction
        opt.verbose && println("- Image reconstruction...")
        flag_interp ? (θ_ = reshape(Ip*vec(θ), nt, 6)) : (θ_ = θ)
        u, fval_recon[n] = image_reconstruction(F(θ_), d, u, set_reg_mean(opt.image_reconstruction, u0))

        # Motion-parameter estimation
        (n == opt.niter) && break
        opt.verbose && println("- Motion-parameter estimation...")
        θ, fval_parest[n] = parameter_estimation(F, u, d, θ, opt.parameter_estimation)

    end

    return u, θ, Dict(:image_reconstruction => fval_recon, :parameter_estimation => fval_parest)

end