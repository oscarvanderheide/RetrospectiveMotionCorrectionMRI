# Miscellanea

export psnr, ssim, differential_interpolation_time

psnr(u_noisy::AbstractArray{T,2}, u_ref::AbstractArray{T,2}) where T = assess_psnr(abs.(u_noisy), abs.(u_ref))
ssim(u_noisy::AbstractArray{T,2}, u_ref::AbstractArray{T,2}) where T = assess_ssim(abs.(u_noisy), abs.(u_ref))
psnr(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3}; preproc::Function=x->abs.(x)) where T = assess_psnr(preproc(u_noisy), preproc(u_ref))
ssim(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3}; preproc::Function=x->abs.(x)) where T = assess_ssim(preproc(u_noisy), preproc(u_ref))

function differential_interpolation_time(n::Integer, σ::T; res_fact::Integer=10) where {T<:Real}
    nx = 2^res_fact
    h = T(2)/(nx-1)
    x = σ*randn(T, n, 2)
    x = round.(x[(abs.(x[:,1]) .< 1) .&& (abs.(x[:,2]) .< 1),:]/h)
    ordering = x[:,1]+x[:,2]*(nx-1)
    ordering .-= minimum(ordering)
    ordering ./= maximum(ordering)
    return sort(ordering)
end