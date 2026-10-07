# Toeplitz embedding of the NFFT normal operator

export ToeplitzNormalOperator, toeplitz_normal_operator


## Normal operator F'*F via circulant embedding
#
# For F = c*diag(phase)*A, with A the type-2 NUFFT and |phase| = 1, the normal operator is a
# (multilevel) Toeplitz convolution:
#   (F'F u)_n = c^2*sum_m T[n-m]*u_m,   T[l] = sum_j exp(i*(k_j.*h)·l),   l ∈ [-(N-1), N-1]
# The kernel T is computed once with a type-1 NUFFT on a 2N grid, after which F'F can be applied
# with two FFTs of size 2N (per dimension) and no NUFFT.

struct ToeplitzNormalOperator{T<:Real}<:AbstractLinearOperator{Complex{T},3,Complex{T},3}
    n::NTuple{3,Int64}
    λ::AbstractArray{Complex{T},3}      # (scaled) eigenvalues of the circulant embedding
    buffer::AbstractArray{Complex{T},3} # 2N workspace
    plan_fwd::Any
    plan_bwd::Any
end

"""
    toeplitz_normal_operator(F::StructuredNFFTtype2LinOp)

Returns the normal operator `F'*F` as a linear operator evaluated via Toeplitz embedding, i.e. with two FFTs on a twice-oversampled grid instead of a type-2 and a type-1 NUFFT. Construction costs one type-1 NUFFT onto the oversampled grid (always computed on the CPU). Memory: two complex arrays of size `2 .*size(F.spatial_geometry)`, on the same device as `F`.
"""
function toeplitz_normal_operator(F::StructuredNFFTtype2LinOp{T}) where {T<:Real}
    n = size(F.spatial_geometry)
    h = spacing(F.spatial_geometry)
    m = 2 .*n
    # Kernel T[l], l = -N..N-1 (FINUFFT mode ordering)
    k = Array(F.kcoord)
    kernel = nufft3d1(vec(k[:,:,1]*h[1]), vec(k[:,:,2]*h[2]), vec(k[:,:,3]*h[3]), ones(Complex{T}, length(F.phase_shift)), 1, F.tol, m...)[:,:,:,1]
    # FFT plans on the device of F
    buffer = similar(F.phase_shift, Complex{T}, m)
    plan_fwd, plan_bwd = inplace_fft_plans(buffer)
    # Eigenvalues of the circulant embedding (index 1 <-> l = 0), including the operator scaling
    # and the normalization of the backward FFT
    copyto!(buffer, ifftshift(kernel)); plan_fwd*buffer; λ = copy(buffer)
    λ .*= F.norm_constant^2/prod(m)
    return ToeplitzNormalOperator{T}(n, λ, buffer, plan_fwd, plan_bwd)
end

# FFTW.MEASURE overwrites the array while planning (so plan before filling it). Planning is slow the
# first time for a given size, but FFTW reuses its "wisdom" afterwards
inplace_fft_plans(b::Array) = (plan_fft!(b; flags=FFTW.MEASURE, num_threads=Threads.nthreads()), plan_bfft!(b; flags=FFTW.MEASURE, num_threads=Threads.nthreads()))
inplace_fft_plans(b::AbstractArray) = (plan_fft!(b), plan_bfft!(b))

AbstractLinearOperators.domain_size(N::ToeplitzNormalOperator) = N.n
AbstractLinearOperators.range_size(N::ToeplitzNormalOperator) = N.n
AbstractLinearOperators.label(::ToeplitzNormalOperator) = "F'F (Toeplitz)"

function AbstractLinearOperators.matvecprod(N::ToeplitzNormalOperator{T}, u::AbstractArray{Complex{T},3}) where {T<:Real}
    b = N.buffer
    zeropad!(b, u)
    N.plan_fwd*b
    pointwise_mult!(b, N.λ)
    N.plan_bwd*b
    return b[1:N.n[1], 1:N.n[2], 1:N.n[3]]
end

zeropad!(b::AbstractArray{CT,3}, u::AbstractArray{CT,3}) where {CT} = (fill!(b, 0); view(b, axes(u)...) .= u; b)
pointwise_mult!(b::AbstractArray{CT,3}, λ::AbstractArray{CT,3}) where {CT} = (b .*= λ)

function zeropad!(b::Array{CT,3}, u::Array{CT,3}) where {CT}
    n1, n2, n3 = size(u)
    Threads.@threads for k = 1:size(b, 3)
        @inbounds for j = 1:size(b, 2), i = 1:size(b, 1)
            b[i,j,k] = (i <= n1 && j <= n2 && k <= n3) ? u[i,j,k] : zero(CT)
        end
    end
    return b
end

function pointwise_mult!(b::Array{CT,3}, λ::Array{CT,3}) where {CT}
    Threads.@threads for k = 1:size(b, 3)
        @inbounds for j = 1:size(b, 2), i = 1:size(b, 1)
            b[i,j,k] *= λ[i,j,k]
        end
    end
    return b
end

AbstractLinearOperators.matvecprod_adj(N::ToeplitzNormalOperator{T}, u::AbstractArray{Complex{T},3}) where {T<:Real} = matvecprod(N, u) # self-adjoint
