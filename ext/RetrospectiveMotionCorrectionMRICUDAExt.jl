# GPU support (CUDA), loaded when CUDA.jl, cuDNN.jl (GPU convolutions via NNlib) and NonuniformFFTs.jl are loaded.
#
# Usage: move the NUFFT operator, data, initial image and reference image to the GPU with `cu`;
# motion parameters stay on the CPU:
#
#   using CUDA, cuDNN, NonuniformFFTs, RetrospectiveMotionCorrectionMRI
#   F = cu(nfft_linop(X, K; tol=1f-4)); d = cu(d); u0 = cu(u0); ref = cu(ref)

module RetrospectiveMotionCorrectionMRICUDAExt

using RetrospectiveMotionCorrectionMRI, CUDA, NonuniformFFTs
using RetrospectiveMotionCorrectionMRI.AbstractLinearOperators: domain_size
const UMRI = RetrospectiveMotionCorrectionMRI.UtilitiesForMRI
const FSW = RetrospectiveMotionCorrectionMRI.FastSolversForWeightedTV
using RetrospectiveMotionCorrectionMRI.FastSolversForWeightedTV: WeightedGradientOperator, convert_spacing


## NUFFT (NonuniformFFTs.jl)

# Kernel half-width giving (at least) the requested relative accuracy, with oversampling σ=2
# (measured relative errors: m=2: 2e-3, m=3: 2e-5, m=4: 8e-6 (Float32 floor))
nufft_halfsupport(tol::Real) = clamp(ceil(Int, 0.5-log10(tol)/2), 2, 8)

# Plans are cached in the operator (and shared with operators derived from it, e.g. F(θ))
function nufft_plan(F::StructuredNFFTtype2LinOp{T}) where {T<:Real}
    n = domain_size(F)
    m = nufft_halfsupport(F.tol)
    F.cache[] isa Dict || (F.cache[] = Dict{Any,Any}())
    return get!(F.cache[], (Complex{T}, n, m)) do
        PlanNUFFT(Complex{T}, n; m=HalfSupport(m), σ=2.0, fftshift=true, backend=CUDABackend())
    end
end

# NonuniformFFTs uses the opposite sign convention (type 2: exp(+ikx)), hence the points -k.*h
function set_kspace_points!(plan, F::StructuredNFFTtype2LinOp{T}, k::CuArray{T,3}) where {T<:Real}
    h = spacing(F.spatial_geometry)
    set_points!(plan, ntuple(d -> fold_unit_cell.(vec(k[:,:,d]).*(-T(h[d]))), 3))
    return plan
end

# Fold points onto [0, 2π). NonuniformFFTs folds points itself, but on the GPU (v0.9.7) tiny
# negative values (e.g. -1f-8, which occur for rotated k = 0 coordinates) are mapped to 2π+x,
# which rounds to exactly 2π and leads to out-of-bounds memory accesses (and wrong results or
# crashes). Points already in [0, 2π) are left untouched by NonuniformFFTs.
@inline function fold_unit_cell(x::T) where {T<:AbstractFloat}
    L = T(2π)
    y = mod(x, L)
    return ifelse(y >= L, zero(T), y)
end

function UMRI.nufft_type2(F::StructuredNFFTtype2LinOp{T}, k::CuArray{T,3}, u::AbstractArray{Complex{T},3}) where {T<:Real}
    plan = set_kspace_points!(nufft_plan(F), F, k)
    d = CuVector{Complex{T}}(undef, size(k, 1)*size(k, 2))
    exec_type2!(d, plan, u isa CuArray ? u : CuArray(u))
    return d
end

function UMRI.nufft_type1(F::StructuredNFFTtype2LinOp{T}, k::CuArray{T,3}, c::AbstractVector{Complex{T}}) where {T<:Real}
    plan = set_kspace_points!(nufft_plan(F), F, k)
    u = CuArray{Complex{T}}(undef, domain_size(F))
    exec_type1!(u, plan, c isa CuArray ? c : CuArray(c))
    return u
end



## Weighted TV kernels (same arithmetic as the multithreaded CPU kernels in FastSolversForWeightedTV)

const NTHREADS = 256
launch_1d!(kernel, N, args...) = (N > 0 && @cuda threads=NTHREADS blocks=cld(N, NTHREADS) kernel(args...); nothing)

@inline function lin2sub(idx, n1, n2)
    i = (idx-1)%n1+1; r = (idx-1)÷n1
    return i, r%n2+1, r÷n2+1
end

function weighted_gradient_cuda!(p, u, ξ, γ, h1, h2, h3, weighted::Bool)
    nx, ny, nz = size(u)
    idx = (blockIdx().x-1)*blockDim().x+threadIdx().x
    idx > (nx-1)*(ny-1)*(nz-1) && return nothing
    i, j, k = lin2sub(idx, nx-1, ny-1)
    @inbounds begin
        c = u[i,j,k]
        g1 = (u[i+1,j,k]-c)/h1; g2 = (u[i,j+1,k]-c)/h2; g3 = (u[i,j,k+1]-c)/h3
        if weighted
            ξ1 = ξ[i,j,k,1]; ξ2 = ξ[i,j,k,2]; ξ3 = ξ[i,j,k,3]
            s = γ*(g1*conj(ξ1)+g2*conj(ξ2)+g3*conj(ξ3))
            g1 -= ξ1*s; g2 -= ξ2*s; g3 -= ξ3*s
        end
        p[i,j,k,1] = g1; p[i,j,k,2] = g2; p[i,j,k,3] = g3
    end
    return nothing
end

function projection_cuda!(q, p, ξ, γ)
    nx, ny, nz, _ = size(p)
    idx = (blockIdx().x-1)*blockDim().x+threadIdx().x
    idx > nx*ny*nz && return nothing
    i, j, k = lin2sub(idx, nx, ny)
    @inbounds begin
        p1 = p[i,j,k,1]; p2 = p[i,j,k,2]; p3 = p[i,j,k,3]
        ξ1 = ξ[i,j,k,1]; ξ2 = ξ[i,j,k,2]; ξ3 = ξ[i,j,k,3]
        s = γ*(p1*conj(ξ1)+p2*conj(ξ2)+p3*conj(ξ3))
        q[i,j,k,1] = p1-ξ1*s; q[i,j,k,2] = p2-ξ2*s; q[i,j,k,3] = p3-ξ3*s
    end
    return nothing
end

function neg_divergence_cuda!(u, q, h1, h2, h3)
    nx, ny, nz = size(u)
    idx = (blockIdx().x-1)*blockDim().x+threadIdx().x
    idx > nx*ny*nz && return nothing
    i, j, k = lin2sub(idx, nx, ny)
    in_i = i < nx; in_j = j < ny; in_k = k < nz
    acc = zero(eltype(u))
    @inbounds begin
        (in_i && in_j && in_k) && (acc -= q[i,j,k,1]/h1+q[i,j,k,2]/h2+q[i,j,k,3]/h3)
        (i > 1 && in_j && in_k) && (acc += q[i-1,j,k,1]/h1)
        (j > 1 && in_i && in_k) && (acc += q[i,j-1,k,2]/h2)
        (k > 1 && in_i && in_j) && (acc += q[i,j,k-1,3]/h3)
        u[i,j,k] = acc
    end
    return nothing
end

function FSW.weighted_gradient!(p::CuArray{T,4}, A::WeightedGradientOperator{T,3,4}, u::CuArray{T,3}) where {T}
    h1, h2, h3 = convert_spacing(T, A.h)
    nx, ny, nz = size(u)
    weighted = ~isnothing(A.weight)
    ξ = weighted ? A.weight.ξ : p # (unused if not weighted)
    ξ isa CuArray{T,4} || return invoke(FSW.weighted_gradient!, Tuple{AbstractArray{T,4},WeightedGradientOperator{T,3,4},AbstractArray{T,3}}, p, A, u)
    γ = weighted ? A.weight.γ : zero(T)
    launch_1d!(weighted_gradient_cuda!, (nx-1)*(ny-1)*(nz-1), p, u, ξ, γ, h1, h2, h3, weighted)
    return p
end

function FSW.weighted_gradient_adj!(u::CuArray{T,3}, A::WeightedGradientOperator{T,3,4}, p::CuArray{T,4}) where {T}
    h1, h2, h3 = convert_spacing(T, A.h)
    if isnothing(A.weight)
        q = p
    else
        ξ = A.weight.ξ
        ξ isa CuArray{T,4} || return invoke(FSW.weighted_gradient_adj!, Tuple{AbstractArray{T,3},WeightedGradientOperator{T,3,4},AbstractArray{T,4}}, u, A, p)
        q = A.buffer[]
        (q isa CuArray{T,4} && size(q) == size(p)) || (q = similar(p); A.buffer[] = q)
        launch_1d!(projection_cuda!, prod(size(p)[1:3]), q, p, ξ, A.weight.γ)
    end
    launch_1d!(neg_divergence_cuda!, length(u), u, q, h1, h2, h3)
    return u
end

function ball_norms_cuda!(ptn, p, G, L, η2)
    nx, ny, nz = size(ptn)
    idx = (blockIdx().x-1)*blockDim().x+threadIdx().x
    idx > nx*ny*nz && return nothing
    i, j, k = lin2sub(idx, nx, ny)
    @inbounds begin
        w1 = L*(p[i,j,k,1]-G[i,j,k,1]/L); w2 = L*(p[i,j,k,2]-G[i,j,k,2]/L); w3 = L*(p[i,j,k,3]-G[i,j,k,3]/L)
        ptn[i,j,k] = sqrt(abs2(w1)+abs2(w2)+abs2(w3)+η2)
    end
    return nothing
end

function ball_update_cuda!(p_, p, G, L, ptn, λ)
    nx, ny, nz = size(ptn)
    idx = (blockIdx().x-1)*blockDim().x+threadIdx().x
    idx > nx*ny*nz && return nothing
    i, j, k = lin2sub(idx, nx, ny)
    @inbounds begin
        s = (ptn[i,j,k] >= λ) ? (1-λ/ptn[i,j,k]) : zero(λ)
        for d = 1:3
            z = p[i,j,k,d]-G[i,j,k,d]/L
            p_[i,j,k,d] = z-s*(L*z)/L
        end
    end
    return nothing
end

function FSW.conjugate_ball_prox!(p_::CuArray{CT,4}, p::CuArray{CT,4}, G::CuArray{CT,4}, L::T, ε::T, ptn::CuArray{T,3}) where {T<:Real,CT<:Union{T,Complex{T}}}
    launch_1d!(ball_norms_cuda!, length(ptn), ptn, p, G, L, 3*eps(T)^2)
    λ = FSW.l1ball_threshold(ptn, ε)
    launch_1d!(ball_update_cuda!, length(ptn), p_, p, G, L, ptn, λ)
    return p_
end


## Gauss-Newton blocks: GN[t,i,j] = real(sum_k conj(A[t,k,i])*B[t,k,j]) (one thread per t)

function gauss_newton_blocks_cuda!(GN, A, B, symmetric::Bool)
    nt, nk, _ = size(A)
    t = (blockIdx().x-1)*blockDim().x+threadIdx().x
    t > nt && return nothing
    @inbounds for i = 1:6, j = (symmetric ? i : 1):6
        acc = zero(eltype(GN))
        for k = 1:nk
            acc += real(conj(A[t,k,i])*B[t,k,j])
        end
        GN[t,i,j] = acc
        symmetric && (GN[t,j,i] = acc)
    end
    return nothing
end

function UMRI.gauss_newton_blocks(A::CuArray{Complex{T},3}, B::CuArray{Complex{T},3}; symmetric::Bool=false) where {T<:Real}
    GN = CuArray{T,3}(undef, size(A, 1), 6, 6)
    launch_1d!(gauss_newton_blocks_cuda!, size(A, 1), GN, A, B, symmetric)
    return GN
end

end
