# Miscellanea

export psnr, ssim

psnr(u_noisy::AbstractArray{T,N}, u_ref::AbstractArray{T,N}) where {T,N} = abs(assess_psnr(abs.(u_ref), abs.(u_noisy)))
ssim(u_noisy::AbstractArray{T,N}, u_ref::AbstractArray{T,N}) where {T,N} = abs(assess_ssim(abs.(u_ref), abs.(u_noisy)))