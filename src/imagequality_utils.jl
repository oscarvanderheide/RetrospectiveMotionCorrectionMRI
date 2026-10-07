# Image-quality metrics

export psnr, ssim

psnr(u_noisy::AbstractArray{T,2}, u_ref::AbstractArray{T,2}) where {T<:Real} = assess_psnr(u_noisy, u_ref)

"""
    psnr(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3};
         slices::Union{FullVolume,NTuple{N,VolumeSlice}}=full_volume(),
         orientation::Orientation=standard_orientation()) where {T<:Real,N}

Compute 2D/3D peak signal-to-noise ratio for the indicated 2D slices of a 3D array or full volume. The optional keyword `slices` indicates the 2D slices in object (see [`volume_slice`](@ref)), according to the 3D `orientation` (see [`orientation`](@ref)).
"""
function psnr(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3}; slices::Union{FullVolume,FullVolume,NTuple{N,VolumeSlice}}=full_volume(), orientation::Orientation=standard_orientation()) where {T<:Real,N}
    (slices isa FullVolume) && (return assess_psnr(u_noisy, u_ref))

    psnr_vec = Vector{T}(undef, length(slices))
    @inbounds for (n, slice) = enumerate(slices)
        u_noisy_slice = select(u_noisy, slice; orientation=orientation)
        u_ref_slice = select(u_ref, slice; orientation=orientation)
        psnr_vec[n] = psnr(u_noisy_slice, u_ref_slice)
    end
    return tuple(psnr_vec...)
end

ssim(u_noisy::AbstractArray{T,2}, u_ref::AbstractArray{T,2}) where {T<:Real} = assess_ssim(u_noisy, u_ref)

"""
    ssim(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3};
         slices::Union{FullVolume,NTuple{N,VolumeSlice}}=full_volume,
         orientation::Orientation=standard_orientation()) where {T<:Real,N}

Compute 2D/3D structural similarity index for the indicated 2D slices of a 3D array or full volume. The optional keyword `slices` indicates the 2D slices in object (see [`volume_slice`](@ref)), according to the 3D `orientation` (see [`orientation`](@ref)).
"""
function ssim(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3}; slices::Union{FullVolume,NTuple{N,VolumeSlice}}=full_volume(), orientation::Orientation=standard_orientation()) where {T<:Real,N}
    (slices isa FullVolume) && (return assess_ssim(u_noisy, u_ref))

    ssim_vec = Vector{T}(undef, length(slices))
    @inbounds for (n, slice) = enumerate(slices)
        u_noisy_slice = select(u_noisy, slice; orientation=orientation)
        u_ref_slice = select(u_ref, slice; orientation=orientation)
        ssim_vec[n] = ssim(u_noisy_slice, u_ref_slice)
    end
    return tuple(ssim_vec...)
end