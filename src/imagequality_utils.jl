# Image-quality metrics

export psnr, ssim

psnr(u_noisy::AbstractArray{T,2}, u_ref::AbstractArray{T,2}) where {T<:Real} = assess_psnr(u_noisy, u_ref)
function psnr(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3}; slices::Union{Nothing,NTuple{N,VolumeSlice}}=nothing, orientation::Orientation=standard_orientation()) where {T<:Real,N}
    if isnothing(slices)
        x, y, z = div.(size(u_noisy), 2).+1
        slices = (VolumeSlice(1, x), VolumeSlice(2, y), VolumeSlice(3, z))
    end
    psnr_vec = Vector{T}(undef, length(slices))
    @inbounds for (n, slice) = enumerate(slices)
        u_noisy_slice = select(u_noisy, slice; orientation=orientation)
        u_ref_slice = select(u_ref, slice; orientation=orientation)
        psnr_vec[n] = psnr(u_noisy_slice, u_ref_slice)
    end
    return tuple(psnr_vec...)
end

ssim(u_noisy::AbstractArray{T,2}, u_ref::AbstractArray{T,2}) where {T<:Real} = assess_ssim(u_noisy, u_ref)
function ssim(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3}; slices::Union{Nothing,NTuple{N,VolumeSlice}}=nothing, orientation::Orientation=standard_orientation()) where {T<:Real,N}
    if isnothing(slices)
        x, y, z = div.(size(u_noisy), 2).+1
        slices = (VolumeSlice(1, x), VolumeSlice(2, y), VolumeSlice(3, z))
    end
    ssim_vec = Vector{T}(undef, length(slices))
    @inbounds for (n, slice) = enumerate(slices)
        u_noisy_slice = select(u_noisy, slice; orientation=orientation)
        u_ref_slice = select(u_ref, slice; orientation=orientation)
        ssim_vec[n] = ssim(u_noisy_slice, u_ref_slice)
    end
    return tuple(ssim_vec...)
end