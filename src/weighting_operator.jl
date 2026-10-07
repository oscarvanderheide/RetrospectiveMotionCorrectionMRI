# Projection operator on vector field

export ProjVectorField, structural_weight, structural_mean, structural_maximum


# Projection on vector field

struct ProjVectorField{T,N}<:AbstractLinearOperator{T,N,N}
    ξ::AbstractArray{T,N}
    γ::T
end

AbstractLinearOperators.domain_size(P::ProjVectorField) = size(P.ξ)
AbstractLinearOperators.range_size(P::ProjVectorField) = size(P.ξ)
AbstractLinearOperators.matvecprod(P::ProjVectorField{T,N}, u::AbstractArray{T,N}) where {T,N} = u-P.γ*(P.ξ.*AbstractProximableFunctions.ptdot(u,P.ξ))
AbstractLinearOperators.matvecprod_adj(P::ProjVectorField{T,N}, u::AbstractArray{T,N}) where {T,N} = P*u

function structural_weight(u::AbstractArray{CT,N}; h::NTuple{N,T}=tuple(ones(T,N)...), γ::T=T(1), η::T=T(0)) where {T<:Real,CT<:RealOrComplex{T},N}
    ∇u = gradient_eval(u, h)
    return ProjVectorField{CT,N+1}(∇u./AbstractProximableFunctions.ptnorm2(∇u; η=η), γ)
end


# Utils

structural_mean(u::AbstractArray{CT,N}; h::NTuple{N,T}=tuple(ones(T,N)...)) where {T<:Real,CT<:RealOrComplex{T},N} = sum(AbstractProximableFunctions.ptnorm2(gradient_eval(u, h)))[1]/prod(size(u))
structural_maximum(u::AbstractArray{CT,N}; h::NTuple{N,T}=tuple(ones(T,N)...)) where {T<:Real,CT<:RealOrComplex{T},N} = norm(AbstractProximableFunctions.ptnorm2(gradient_eval(u, h)), Inf)