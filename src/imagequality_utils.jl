# Image-quality metrics

export psnr, ssim

psnr(u_noisy::AbstractArray{T,N}, u_ref::AbstractArray{T,N}; preproc::Function=x->abs.(x)) where {T,N} = assess_psnr(preproc(u_noisy), preproc(u_ref))
ssim(u_noisy::AbstractArray{T,N}, u_ref::AbstractArray{T,N}; preproc::Function=x->abs.(x)) where {T,N} = assess_ssim(preproc(u_noisy), preproc(u_ref))