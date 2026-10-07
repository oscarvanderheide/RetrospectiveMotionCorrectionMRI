#: Abstract types
export AbstractLinearOperator, AbstractAutoLinearOperator,
       domain_size, range_size, domain_eltype, range_eltype,
       label,
       matvecprod!, matvecprod_adj!, invmatvecprod!, invmatvecprod_adj!,
       matvecprod, matvecprod_adj, invmatvecprod, invmatvecprod_adj,
       full_matrix

abstract type AbstractLinearOperator{TD<:Number,ND,TR<:Number,NR} end
abstract type AbstractAutoLinearOperator{T<:Number,N}<:AbstractLinearOperator{T,N,T,N} end


# Base functions

domain_size(::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = Tuple(Vector{Nothing}(undef,ND))
range_size( ::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = Tuple(Vector{Nothing}(undef,NR))
domain_eltype(::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = TD
range_eltype( ::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = TR

label(::AbstractLinearOperator) = nothing

matvecprod!(       ::AbstractArray{TD,ND}, ::AbstractLinearOperator{TD,ND,TR,NR}, ::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = throw(ArgumentError("Product not implemented for this operator"))
matvecprod_adj!(   ::AbstractArray{TD,ND}, ::AbstractLinearOperator{TD,ND,TR,NR}, ::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = throw(ArgumentError("Adjoint product not implemented for this operator"))
invmatvecprod!(    ::AbstractArray{TD,ND}, ::AbstractLinearOperator{TD,ND,TR,NR}, ::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = throw(ArgumentError("Inverse product not implemented for this operator"))
invmatvecprod_adj!(::AbstractArray{TD,ND}, ::AbstractLinearOperator{TD,ND,TR,NR}, ::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = throw(ArgumentError("Adjoint inverse product not implemented for this operator"))

matvecprod(A::AbstractLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = matvecprod!(similar(u, range_size(A)), A, u)
matvecprod_adj(A::AbstractLinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = matvecprod_adj!(similar(v, domain_size(A)), A, v)
invmatvecprod(A::AbstractLinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = invmatvecprod!(similar(v, range_size(A)), A, v)
invmatvecprod_adj(A::AbstractLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = invmatvecprod_adj!(similar(u, domain_size(A)), A, u)


# Algebra

Base.:*(A::AbstractLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = matvecprod(A, u)
Base.:\(A::AbstractLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = invmatvecprod(A, u)


# Display behaviour

Base.show(::IO, A::AbstractLinearOperator) = info(A)
Base.show(::IO, mime::MIME"text/plain", A::AbstractLinearOperator) = info(A)
info(A::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = print("Linear operator, domain ≅ ", TD, "^", domain_size(A), ", range ≅ ", TR, "^", range_size(A), ~isnothing(label(A)) ? string(", label=", label(A)) : "")


# Other utils

Base.size(A::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = ((TR, range_size(A)...), (TD, domain_size(A)...))