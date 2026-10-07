# NFFT utilities

export StructuredNFFTtype2LinOp, ParametericStructuredNFFTtype2, StructuredNFFTtype2DelayedEval, JacobianStructuredNFFTtype2
export nfft_linop, Jacobian, ∂, sparse_matrix_GaussNewton


## General type-2 NFFT linear operator

struct StructuredNFFTtype2LinOp{T<:Real}<:AbstractNFFTLinOp{T,AbstractCartesianSpatialGeometry{T},AbstractStructuredKSpaceSampling{T},3,2}
    spatial_geometry::AbstractCartesianSpatialGeometry{T}
    kcoord::AbstractArray{T,3}
    phase_shift::AbstractArray{Complex{T},2}
    norm_constant::T
    tol::T
    cache::Base.RefValue{Any} # backend-specific workspace (e.g. GPU NUFFT plans), shared with derived operators
end

StructuredNFFTtype2LinOp{T}(X, K, phase_shift, norm_constant, tol) where {T<:Real} = StructuredNFFTtype2LinOp{T}(X, K, phase_shift, norm_constant, tol, Ref{Any}(nothing))

function nfft_linop(X::CartesianSpatialGeometry{T}, K::AbstractArray{T,3}; norm_constant::T=1/T(sqrt(prod(X.nsamples))), tol::T=T(1e-6), cache::Base.RefValue{Any}=Ref{Any}(nothing)) where {T<:Real}
    o = origin(X; wrt_center=true)
    phase_shift = exp.(im*(K[:,:,1]*o[1]+K[:,:,2]*o[2]+K[:,:,3]*o[3]))
    return StructuredNFFTtype2LinOp{T}(X, K, phase_shift, norm_constant, tol, cache)
end

# Move CPU data (e.g. motion parameters, coordinate vectors) to the device where `ref` lives
on_device_of(ref::AbstractArray, A::AbstractArray) = copyto!(similar(ref, eltype(A), size(A)), collect(A))
on_device_of(::Array, A::Array) = A

# GPU support: `cu(F)` (or `adapt(CuArray, F)`) moves the operator to the GPU (requires the CUDA extension)
Adapt.adapt_structure(to, F::StructuredNFFTtype2LinOp{T}) where {T<:Real} = StructuredNFFTtype2LinOp{T}(F.spatial_geometry, adapt(to, F.kcoord), adapt(to, F.phase_shift), F.norm_constant, F.tol, Ref{Any}(nothing))

"""
    nfft_linop(X::CartesianSpatialGeometry, K::StructuredKSpaceSampling)

Return the non-uniform Fourier transform as a linear operator for a specified Cartesian spatial discretization `X` and a ``k``-space trajectory `K`.

## Example

```julia
X = spatial_geometry((1f0, 1f0, 1f0), (32, 32, 32))
K = kspace_sampling(X, (1, 2))
F = nfft_linop(X, K)
u = randn(ComplexF32, X.nsamples)
d = F*u  # evaluation
    F'*d # adjoint
```
"""
nfft_linop(X::CartesianSpatialGeometry{T}, K::AbstractStructuredKSpaceSampling{T}; norm_constant::T=1/T(sqrt(prod(X.nsamples))), tol::T=T(1e-6)) where {T<:Real} = nfft_linop(X, coord(K); norm_constant=norm_constant, tol=tol)

AbstractLinearOperators.domain_size(F::StructuredNFFTtype2LinOp) = size(F.spatial_geometry)
AbstractLinearOperators.range_size(F::StructuredNFFTtype2LinOp) = size(F.phase_shift)

AbstractLinearOperators.matvecprod(F::StructuredNFFTtype2LinOp{T}, u::AbstractArray{Complex{T},3}) where {T<:Real} = F.phase_shift.*reshape(nufft_type2(F, F.kcoord, u), range_size(F))*F.norm_constant

AbstractLinearOperators.matvecprod_adj(F::StructuredNFFTtype2LinOp{T}, d::AbstractArray{Complex{T},2}) where {T<:Real} = nufft_type1(F, F.kcoord, vec(conj(F.phase_shift).*d))*F.norm_constant

# NUFFT backends, dispatched on where the k-space coordinates live (CPU: FINUFFT; GPU: see the CUDA extension)
#   type 2: d_j = sum_n u_n exp(-i(k_j.*h)⋅n),   type 1: u_n = sum_j c_j exp(+i(k_j.*h)⋅n)
function nufft_type2(F::StructuredNFFTtype2LinOp{T}, k::Array{T,3}, u::AbstractArray{Complex{T},3}) where {T<:Real}
    h = spacing(F.spatial_geometry)
    return nufft3d2(vec(k[:,:,1]*h[1]), vec(k[:,:,2]*h[2]), vec(k[:,:,3]*h[3]), -1, F.tol, u)
end

function nufft_type1(F::StructuredNFFTtype2LinOp{T}, k::Array{T,3}, c::AbstractVector{Complex{T}}) where {T<:Real}
    h = spacing(F.spatial_geometry)
    return nufft3d1(vec(k[:,:,1]*h[1]), vec(k[:,:,2]*h[2]), vec(k[:,:,3]*h[3]), c, 1, F.tol, domain_size(F)...)[:,:,:,1]
end


## Rigid-body motion perturbation of NFFT

function (F::StructuredNFFTtype2LinOp{T})(θ::AbstractArray{T,2}) where {T<:Real}
    θ = on_device_of(F.kcoord, θ)
    τ = θ[:,1:3]
    φ = θ[:,4:6]
    k = F.kcoord
    Rφk = rotation_linop(φ)*k
    o = origin(F.spatial_geometry; wrt_center=true)
    phase_shift = exp.(-im*( k[:,:,1].*τ[:,1]+k[:,:,2].*τ[:,2]+k[:,:,3].*τ[:,3]
                            -Rφk[:,:,1]*o[1] -Rφk[:,:,2]*o[2] -Rφk[:,:,3]*o[3]))
    return StructuredNFFTtype2LinOp{T}(F.spatial_geometry, Rφk, phase_shift, F.norm_constant, F.tol, F.cache)
end


## Rigid-body motion parameteric perturbation of NFFT (functional)

struct ParametericStructuredNFFTtype2{T<:Real}
    unperturbed::StructuredNFFTtype2LinOp{T}
end

(F::StructuredNFFTtype2LinOp{T})() where {T<:Real} = ParametericStructuredNFFTtype2{T}(F)

(F::ParametericStructuredNFFTtype2{T})(θ::AbstractArray{T,2}) where {T<:Real} = F.unperturbed(θ)


## Parameteric delayed evaluation

struct StructuredNFFTtype2DelayedEval{T<:Real}
    parameteric_linop::ParametericStructuredNFFTtype2{T}
    input::AbstractArray{Complex{T},3}
end

Base.:*(F::ParametericStructuredNFFTtype2{T}, u::AbstractArray{Complex{T},3}) where {T<:Real} = StructuredNFFTtype2DelayedEval{T}(F, u)

(Fu::StructuredNFFTtype2DelayedEval{T})(θ::AbstractArray{T,2}) where {T<:Real} = Fu.parameteric_linop(θ)*Fu.input


## Jacobian of nfft evaluated

struct JacobianStructuredNFFTtype2{T<:Real}<:AbstractLinearOperator{Complex{T},2,Complex{T},2}
    ∂F::AbstractArray{Complex{T},3}
end

function Jacobian(Fu::StructuredNFFTtype2DelayedEval{T}, θ::AbstractArray{T,2}) where {T<:Real}

    # Simplifying notation
    u = Fu.input
    F = Fu.parameteric_linop.unperturbed
    X = F.spatial_geometry
    K = F.kcoord
    tol = F.tol
    norm_constant = F.norm_constant
    θ = on_device_of(K, θ)
    τ = θ[:,1:3]
    φ = θ[:,4:6]
    P = phase_shift(K) # phase-shift
    R = rotation() # rotation

    # NFFT of u and derivatives thereof
    Kφ, ∂Kφ = ∂(R()*K, φ)
    Fφ = nfft_linop(X, Kφ; tol=tol, norm_constant=norm_constant, cache=F.cache)
    x, y, z = map(c -> on_device_of(K, c), coord(X; mesh=false))
    Fu = Fφ*u
    ∇Fu = -im*cat(Fφ*(u.*reshape(x,:,1,1)), Fφ*(u.*reshape(y,1,:,1)), Fφ*(u.*reshape(z,1,1,:)); dims=3)

    # Computing rigid-body motion Jacobian
    d, Pτ, ∂Pτu = ∂(P()*Fu, τ)
    J = cat(∂Pτu.∂P, Pτ*dot(∇Fu, ∂Kφ); dims=3)

    return d, Pτ*Fφ, JacobianStructuredNFFTtype2{T}(J)

end

∂(Fu::StructuredNFFTtype2DelayedEval{T}, θ::AbstractArray{T,2}) where {T<:Real} = Jacobian(Fu, θ)

AbstractLinearOperators.domain_size(∂Fu::JacobianStructuredNFFTtype2) = size(∂Fu.∂F)[[1,3]]
AbstractLinearOperators.range_size(∂Fu::JacobianStructuredNFFTtype2) = size(∂Fu.∂F)[1:2]
function AbstractLinearOperators.matvecprod(∂Fu::JacobianStructuredNFFTtype2{T}, Δθ::AbstractArray{Complex{T},2}) where {T<:Real}
    Δθ = on_device_of(∂Fu.∂F, Δθ)
    # JΔθ = similar(Δθ, size(∂Fu.∂F, 1), size(∂Fu.∂F, 2))
    # fill!(JΔθ, T(0))
    # @inbounds for i = 1:6
    #     JΔθ .+= ∂Fu.∂F[:,:,i].*Δθ[:,i]
    # end
    # return JΔθ
    return sum(∂Fu.∂F.*reshape(Δθ,:,1,6); dims=3)[:,:,1]
end
Base.:*(∂Fu::JacobianStructuredNFFTtype2{T}, Δθ::AbstractArray{T,2}) where {T<:Real} = ∂Fu*complex(Δθ)
AbstractLinearOperators.matvecprod_adj(∂Fu::JacobianStructuredNFFTtype2{T}, Δd::AbstractArray{Complex{T},2}) where {T<:Real} = real(sum(conj(∂Fu.∂F).*Δd; dims=2)[:,1,:])
function AbstractLinearOperators.matvecprod_adj(∂Fu::JacobianStructuredNFFTtype2{T}, Δd::Array{Complex{T},2}) where {T<:Real}
    J = ∂Fu.∂F
    J isa Array || return real(sum(conj(J).*Δd; dims=2)[:,1,:])
    nt, nk, np = size(J)
    g = Array{T,2}(undef, nt, np)
    Threads.@threads for i = 1:np
        acc = zeros(Float64, nt)
        @inbounds for k = 1:nk, t = 1:nt
            acc[t] += real(conj(J[t,k,i])*Δd[t,k])
        end
        g[:,i] .= acc
    end
    return g
end


## Other utilities

function sparse_matrix_GaussNewton(∂F::JacobianStructuredNFFTtype2{T}; W::Union{Nothing,AbstractLinearOperator}=nothing, H::Union{Nothing,AbstractLinearOperator}=nothing) where {T<:Real}
    J = ∂F.∂F
    if ~isnothing(W)
        WJ = similar(J)
        @inbounds for i = 1:6
            WJ[:,:,i] .= W*J[:,:,i]
        end
    else
        WJ = J
    end
    if ~isnothing(H)
        HWJ = similar(WJ)
        @inbounds for i = 1:6
            HWJ[:,:,i] .= H*WJ[:,:,i]
        end
    else
        HWJ = WJ
    end
    GN = Array(gauss_newton_blocks(WJ, HWJ; symmetric=isnothing(H))) # small (nt×6×6): the sparse Hessian lives on the CPU
    # Block (i,j) of the Hessian is diagm(GN[:,i,j])
    nt = size(J, 1)
    rows = vec([(i-1)*nt+t for t = 1:nt, i = 1:6, j = 1:6])
    cols = vec([(j-1)*nt+t for t = 1:nt, i = 1:6, j = 1:6])
    return sparse(rows, cols, vec(GN), 6*nt, 6*nt)
end

# GN[t,i,j] = real(sum_k conj(A[t,k,i])*B[t,k,j])
function gauss_newton_blocks(A::AbstractArray{Complex{T},3}, B::AbstractArray{Complex{T},3}; symmetric::Bool=false) where {T<:Real}
    GN = similar(A, T, size(A, 1), 6, 6)
    @inbounds for i = 1:6, j = (symmetric ? i : 1):6
        GN[:,i,j] = vec(real(sum(conj(A[:,:,i]).*B[:,:,j]; dims=2)))
        symmetric && (GN[:,j,i] = GN[:,i,j])
    end
    return GN
end

function gauss_newton_blocks(A::Array{Complex{T},3}, B::Array{Complex{T},3}; symmetric::Bool=false) where {T<:Real}
    nt, nk, _ = size(A)
    GN = Array{T,3}(undef, nt, 6, 6)
    pairs = [(i, j) for i = 1:6 for j = (symmetric ? i : 1):6]
    Threads.@threads for p in eachindex(pairs)
        i, j = pairs[p]
        acc = zeros(Float64, nt)
        @inbounds for k = 1:nk, t = 1:nt
            acc[t] += real(conj(A[t,k,i])*B[t,k,j])
        end
        GN[:,i,j] .= acc
        symmetric && (GN[:,j,i] .= acc)
    end
    return GN
end
