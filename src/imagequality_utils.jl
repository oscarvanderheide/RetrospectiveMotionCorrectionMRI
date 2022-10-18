# Image-quality metrics

export psnr, ssim

function psnr(u_noisy::AbstractArray{T,N}, u_ref::AbstractArray{T,N}; slices::Union{Nothing,NTuple{<:Any,VolumeSlice}}, orientation::Orientation=standard_orientation()) where {T<:Real,N}
    if isnothing(slices)
        x, y, z = div.(size(u_noisy), 2).+1
        slices = (VolumeSlice(1, x), VolumeSlice(2, y), VolumeSlice(3, z))
    end
    psnr_slice = similar(u_noisy, length(slices))
    @inbounds for (i, slice) = enumerate(slices)
        u_noisy_slice = select(u_noisy, slice; orientation=orientation)
        u_ref_slice = select(u_ref, slice; orientation=orientation)
        psnr_slice[i] = assess_psnr(u_noisy_slice, u_ref_slice)
    end
    return tuple(psnr_slice...)
end

function ssim(u_noisy::AbstractArray{T,N}, u_ref::AbstractArray{T,N}; slices::Union{Nothing,NTuple{<:Any,VolumeSlice}}, orientation::Orientation=standard_orientation()) where {T<:Real,N}
    if isnothing(slices)
        x, y, z = div.(size(u_noisy), 2).+1
        slices = (VolumeSlice(1, x), VolumeSlice(2, y), VolumeSlice(3, z))
    end
    ssim_slice = similar(u_noisy, length(slices))
    @inbounds for (i, slice) = enumerate(slices)
        u_noisy_slice = select(u_noisy, slice; orientation=orientation)
        u_ref_slice = select(u_ref, slice; orientation=orientation)
        ssim_slice[i] = assess_ssim(u_noisy_slice, u_ref_slice)
    end
    return tuple(ssim_slice...)
end