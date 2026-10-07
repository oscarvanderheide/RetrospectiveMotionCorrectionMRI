export HaarTransform, Haar_transform


# Haar transform

mutable struct HaarTransform{T,N,M}<:AbstractLinearOperator{T,N,T,M}
    op::AbstractConvolutionOperator{T}
    factor::Union{Nothing,T}
    batch::Bool
end

Haar_transform(::Type{T}, D::Integer; orthogonal::Bool=true, batch::Bool=false) where T = HaarTransform{T,batch ? D+2 : D,batch ? D+2 : D+1}(convolution_operator(Haar_stencil(T,D); stride=2, flipped=false), orthogonal ? nothing : (sqrt(T(2))/2)^D, batch)

AbstractLinearOperators.domain_size(W::HaarTransform) = domain_size(W.op)
AbstractLinearOperators.range_size(W::HaarTransform)  = range_size(W.op)
AbstractLinearOperators.label(::HaarTransform) = "HaarTransform"

function AbstractLinearOperators.matvecprod(W::HaarTransform{T,N,M}, u::AbstractArray{T,N}; inverse::Bool=false) where {T,N,M}

    D = W.batch ? N-2 : N
    W.batch ? ((n..., nc, nb) = size(u)) :
              (n = size(u); nc = 1; nb = 1)
    nc_ = nc*2^D
    Wu = similar(u, div.(n,2)..., nc_, nb)
    idx_spatial = Tuple([Colon() for i = 1:length(n)])
    @inbounds for c = 1:nc
        selectdim(Wu, D+1, c:nc:nc_) .= matvecprod(W.op, u[idx_spatial...,c:c,:])
    end
    ~isnothing(W.factor) && (~inverse ? (Wu .*= W.factor) : (Wu ./= W.factor))
    return W.batch ? Wu : dropdims(Wu; dims=D+2)

end

function AbstractLinearOperators.matvecprod_adj(W::HaarTransform{T,N,M}, Wu::AbstractArray{T,M}; inverse::Bool=false) where {T,N,M}

    D = W.batch ? N-2 : N
    W.batch ? ((n..., nc_, nb) = size(Wu); nc = div(nc_,2^D)) :
              ((n..., nc_) = size(Wu); nb = 1; nc = div(nc_,2^D))
    u = similar(Wu, n.*2..., nc, nb)
    idx_spatial = Tuple([Colon() for i = 1:length(n)])
    @inbounds for c = 1:nc
        selectdim(u, D+1, c:c) .= matvecprod_adj(W.op, Wu[idx_spatial...,c:nc:nc_,:])
    end
    ~isnothing(W.factor) && (~inverse ? (u .*= W.factor) : (u ./= W.factor))
    return W.batch ? u : dropdims(u; dims=(D+1,D+2))

end

AbstractLinearOperators.invmatvecprod(W::HaarTransform{T,N,M}, v::AbstractArray{T,M}) where {T,N,M} = matvecprod_adj(W, v; inverse=true)
AbstractLinearOperators.invmatvecprod_adj(W::HaarTransform{T,N,M}, u::AbstractArray{T,N}) where {T,N,M} = matvecprod(W, u; inverse=true)


# Haar utils

function Haar_stencil(::Type{T}, N::Integer) where T

    # One-dimensional case
    (N == 1) && (return cat([1; 1], [-1; 1]; dims=3)/sqrt(T(2)))

    # Initialize stencil
    H = Array{T,N+2}(undef,2*ones(Int,N)...,1,2^N)

    # Recursive definition
    H_1   = Haar_stencil(T, 1)
    H_Nm1 = Haar_stencil(T, N-1)
    @inbounds for i = 1:2^(N-1), j = 1:2
        selectdim(selectdim(H, N+2, 2^(N-1)*(j-1)+i), N+1, 1) .= H_1[:,1,j].*reshape(selectdim(selectdim(H_Nm1, N+1, i), N, 1),1,2*ones(Int,N-1)...)
    end

    return H

end