# Miscellanea

export psnr, ssim, differential_interpolation_time

psnr(u_noisy::AbstractArray{T,2}, u_ref::AbstractArray{T,2}) where T = assess_psnr(abs.(u_noisy), abs.(u_ref))
ssim(u_noisy::AbstractArray{T,2}, u_ref::AbstractArray{T,2}) where T = assess_ssim(abs.(u_noisy), abs.(u_ref))

function psnr(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3}; x=1, y=nothing, z=nothing) where T
    nx = length(x)
    ny = length(y)
    nz = length(z)
    psnr_vals = Vector{real(T)}(undef, nx+ny+nz)
    for i = 1:nx
        psnr_vals[i] = psnr(u_noisy[i, :, :], u_ref[i, :, :])
    end
    for i = 1:ny
        psnr_vals[i+nx] = psnr(u_noisy[:, i, :], u_ref[:, i, :])
    end
    for i = 1:nz
        psnr_vals[i+nx+ny] = psnr(u_noisy[:, :, i], u_ref[:, :, i])
    end
    return tuple(psnr_vals...)
end
function ssim(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3}; x=1, y=nothing, z=nothing) where T
    nx = length(x)
    ny = length(y)
    nz = length(z)
    ssim_vals = Vector{real(T)}(undef, nx+ny+nz)
    for i = 1:nx
        ssim_vals[i] = ssim(u_noisy[i, :, :], u_ref[i, :, :])
    end
    for i = 1:ny
        ssim_vals[i+nx] = ssim(u_noisy[:, i, :], u_ref[:, i, :])
    end
    for i = 1:nz
        ssim_vals[i+nx+ny] = ssim(u_noisy[:, :, i], u_ref[:, :, i])
    end
    return tuple(ssim_vals...)
end

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