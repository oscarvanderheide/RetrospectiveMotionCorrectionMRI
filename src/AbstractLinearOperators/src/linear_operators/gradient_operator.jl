export GradientOperator, gradient_operator


# Gradient

struct GradientOperator{T,N,M}<:AbstractLinearOperator{T,N,T,M}
    op::AbstractConvolutionOperator{T}
    batch::Bool
end

function gradient_operator(::Type{CT}, h::NTuple{D,T}; batch::Bool=false) where {T<:Real,D,CT<:RealOrComplex{T}}
    stencil_size = Tuple([[2 for i = 1:D]..., 1, D])
    stencil = zeros(CT, stencil_size)
    for i = 1:D
        idx = ones(Int, D)
        stencil[idx..., 1, i] = -1/h[i]
        idx[i] = 2
        stencil[idx..., 1, i] =  1/h[i]
    end
    N = batch ? D+2 : D
    M = batch ? D+2 : D+1
    return GradientOperator{CT,N,M}(convolution_operator(stencil; flipped=true, cdims_onthefly=false), batch)
end

AbstractLinearOperators.domain_size(∇::GradientOperator{T,N,M}) where {T,N,M} = domain_size(∇.op)[1:N]
AbstractLinearOperators.range_size( ∇::GradientOperator{T,N,M}) where {T,N,M} = range_size(∇.op)[1:M]
AbstractLinearOperators.label(::GradientOperator) = "∇"

AbstractLinearOperators.matvecprod(∇::GradientOperator{T,N,M}, u::AbstractArray{T,N}) where {T,N,M} = ∇.batch ? matvecprod(∇.op, u) : dropdims(matvecprod(∇.op, reshape(u, size(u)...,1,1)); dims=N+2)
AbstractLinearOperators.matvecprod_adj(∇::GradientOperator{T,N,M}, v::AbstractArray{T,M}) where {T,N,M} = ∇.batch ? matvecprod_adj(∇.op, v) : dropdims(matvecprod_adj(∇.op, reshape(v, size(v)...,1)); dims=Tuple(M:M+1))

AbstractLinearOperators.full_matrix(G::GradientOperator) = full_matrix(G.op)