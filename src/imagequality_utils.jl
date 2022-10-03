# Image-quality metrics

export psnr#, ssim

psnr(u_noisy::AbstractArray{T,N}, u_ref::AbstractArray{T,N}; preproc::Function=x->abs.(x)) where {T,N} = 10*log10(norm(preproc(u_ref), Inf)^2/(norm(preproc(u_noisy)-preproc(u_ref))^2/prod(size(u_noisy))))
# ssim(u_noisy::AbstractArray{T,N}, u_ref::AbstractArray{T,N}; preproc::Function=x->abs.(x)) where {T,N} = assess_ssim(preproc(u_noisy), preproc(u_ref))