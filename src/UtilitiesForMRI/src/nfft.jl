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
end

function nfft_linop(X::CartesianSpatialGeometry{T}, K::AbstractArray{T,3}; norm_constant::T=1/T(sqrt(prod(X.nsamples))), tol::T=T(1e-6)) where {T<:Real}
    o = origin(X; wrt_center=true)
    phase_shift = exp.(im*sum(K.*reshape([o...],1,1,3);dims=3)[:,:,1])
    return StructuredNFFTtype2LinOp{T}(X, K, phase_shift, norm_constant, tol)
end

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

function AbstractLinearOperators.matvecprod(F::StructuredNFFTtype2LinOp{T}, u::AbstractArray{Complex{T},3}) where {T<:Real}
    h = spacing(F.spatial_geometry)
    return F.phase_shift.*reshape(nufft3d2(vec(F.kcoord[:,:,1]*h[1]), vec(F.kcoord[:,:,2]*h[2]), vec(F.kcoord[:,:,3]*h[3]), -1, F.tol, u), range_size(F))*F.norm_constant
end

function AbstractLinearOperators.matvecprod_adj(F::StructuredNFFTtype2LinOp{T}, d::AbstractArray{Complex{T},2}) where {T<:Real}
    h = spacing(F.spatial_geometry)
    return nufft3d1(vec(F.kcoord[:,:,1]*h[1]), vec(F.kcoord[:,:,2]*h[2]), vec(F.kcoord[:,:,3]*h[3]), vec(conj(F.phase_shift).*d), 1, F.tol, domain_size(F)...)[:,:,:,1]*F.norm_constant
end


## Rigid-body motion perturbation of NFFT

function (F::StructuredNFFTtype2LinOp{T})(θ::AbstractArray{T,2}) where {T<:Real}
    τ = θ[:,1:3]
    φ = θ[:,4:6]
    k = F.kcoord
    Rφk = rotation_linop(φ)*k
    o = origin(F.spatial_geometry; wrt_center=true)
    phase_shift = exp.(-im*( k[:,:,1].*τ[:,1]+k[:,:,2].*τ[:,2]+k[:,:,3].*τ[:,3]
                            -Rφk[:,:,1]*o[1] -Rφk[:,:,2]*o[2] -Rφk[:,:,3]*o[3]))
    return StructuredNFFTtype2LinOp{T}(F.spatial_geometry, Rφk, phase_shift, F.norm_constant, F.tol)
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

struct JacobianStructuredNFFTtype2{T<:Real}<:AbstractLinearOperator{Complex{T},2,2}
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
    τ = θ[:,1:3]
    φ = θ[:,4:6]
    P = phase_shift(K) # phase-shift
    R = rotation() # rotation

    # NFFT of u and derivatives thereof
    Kφ, ∂Kφ = ∂(R()*K, φ)
    Fφ = nfft_linop(X, Kφ; tol=tol, norm_constant=norm_constant)
    x, y, z = coord(X; mesh=false)
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


## Other utilities

function sparse_matrix_GaussNewton(∂F::JacobianStructuredNFFTtype2{T}; W::Union{Nothing,AbstractLinearOperator}=nothing, H::Union{Nothing,AbstractLinearOperator}=nothing) where {T<:Real}
    J = ∂F.∂F
    GN = similar(J, T, size(J, 1), 6, 6)
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
    @inbounds for i = 1:6, j = 1:6
        GN[:,i,j] = vec(real(sum(conj(WJ[:,:,i]).*HWJ[:,:,j]; dims=2)))
    end
    return hvcat(6, [spdiagm(0 => GN[:,i,j]) for j=1:6,i=1:6]...)
end