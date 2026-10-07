# GPU support (CUDA), loaded when both CUDA.jl and NonuniformFFTs.jl are loaded.
#
# Usage: move the NUFFT operator, data, initial image and reference image to the GPU with `cu`;
# motion parameters stay on the CPU:
#
#   using CUDA, NonuniformFFTs, RetrospectiveMotionCorrectionMRI
#   F = cu(nfft_linop(X, K; tol=1f-4)); d = cu(d); u0 = cu(u0); ref = cu(ref)

module RetrospectiveMotionCorrectionMRICUDAExt

using RetrospectiveMotionCorrectionMRI, CUDA, NonuniformFFTs
using RetrospectiveMotionCorrectionMRI.AbstractLinearOperators: domain_size
const UMRI = RetrospectiveMotionCorrectionMRI.UtilitiesForMRI


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

end
