#: Linear algebra for AbstractLinearOperator


# Unary operations

## AdjointLinearOperators: adjoint(A)

struct AdjointLinearOperator{TD,ND,TR,NR}<:AbstractLinearOperator{TD,ND,TR,NR}
    A::AbstractLinearOperator{TR,NR,TD,ND}
end

LinearAlgebra.adjoint(A::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = AdjointLinearOperator{TR,NR,TD,ND}(A)

domain_size(A::AdjointLinearOperator) = range_size(A.A)
range_size(A::AdjointLinearOperator) = domain_size(A.A)
label(A::AdjointLinearOperator) = string("adjoint(", label(A.A), ")")
matvecprod(A::AdjointLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = matvecprod_adj(A.A, u)
matvecprod_adj(A::AdjointLinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = matvecprod(A.A, v)
invmatvecprod(A::AdjointLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = invmatvecprod_adj(A.A, u)
invmatvecprod_adj(A::AdjointLinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = invmatvecprod(A.A, v)

## InverseLinearOperators: inv(A)

struct InverseLinearOperator{TD,ND,TR,NR}<:AbstractLinearOperator{TD,ND,TR,NR}
    A::AbstractLinearOperator{TR,NR,TD,ND}
end

LinearAlgebra.inv(A::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = InverseLinearOperator{TR,NR,TD,ND}(A)

domain_size(Ainv::InverseLinearOperator) = range_size(Ainv.A)
range_size(Ainv::InverseLinearOperator) = domain_size(Ainv.A)
label(Ainv::InverseLinearOperator) = string("inverse(", label(Ainv.A), ")")
matvecprod(Ainv::InverseLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = invmatvecprod(Ainv.A, u)
matvecprod_adj(Ainv::InverseLinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = invmatvecprod_adj(Ainv.A, v)
invmatvecprod(Ainv::InverseLinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = matvecprod(Ainv.A, v)
invmatvecprod_adj(Ainv::InverseLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = matvecprod_adj(Ainv.A, u)

## NegativeOperators: -A

struct NegativeLinearOperator{TD,ND,TR,NR}<:AbstractLinearOperator{TD,ND,TR,NR}
    A::AbstractLinearOperator{TD,ND,TR,NR}
end

Base.:-(A::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = NegativeLinearOperator{TD,ND,TR,NR}(A)

domain_size(A::NegativeLinearOperator) = domain_size(A.A)
range_size(A::NegativeLinearOperator) = range_size(A.A)
label(A::NegativeLinearOperator) = string("-(", label(A.A), ")")
matvecprod(A::NegativeLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = -matvecprod(A.A, u)
matvecprod_adj(A::NegativeLinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = -matvecprod_adj(A.A, v)
invmatvecprod(A::NegativeLinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = -invmatvecprod(A.A, v)
invmatvecprod_adj(A::NegativeLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = -invmatvecprod_adj(A.A, u)


# Binary operations

## PlusLinearOperators: A+B

struct PlusLinearOperator{TD,ND,TR,NR}<:AbstractLinearOperator{TD,ND,TR,NR}
    A::AbstractLinearOperator{TD,ND,TR,NR}
    B::AbstractLinearOperator{TD,ND,TR,NR}
end

Base.:+(A::AbstractLinearOperator{TD,ND,TR,NR}, B::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = PlusLinearOperator{TD,ND,TR,NR}(A, B)

domain_size(A::PlusLinearOperator) = domain_size(A.A)
range_size(A::PlusLinearOperator) = range_size(A.A)
label(A::PlusLinearOperator) = string("(", label(A.A), "+", label(A.B),")")
matvecprod(A::PlusLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = matvecprod(A.A, u)+matvecprod(A.B, u)
matvecprod_adj(A::PlusLinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = matvecprod_adj(A.A, v)+matvecprod_adj(A.B, v)

## MinusLinearOperators: A-B

struct MinusLinearOperator{TD,ND,TR,NR}<:AbstractLinearOperator{TD,ND,TR,NR}
    A::AbstractLinearOperator{TD,ND,TR,NR}
    B::AbstractLinearOperator{TD,ND,TR,NR}
end

Base.:-(A::AbstractLinearOperator{TD,ND,TR,NR}, B::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = MinusLinearOperator{TD,ND,TR,NR}(A, B)

domain_size(A::MinusLinearOperator) = domain_size(A.A)
range_size(A::MinusLinearOperator) = range_size(A.A)
label(A::MinusLinearOperator) = string("(", label(A.A), "-", label(A.B),")")
matvecprod(A::MinusLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = matvecprod(A.A, u)-matvecprod(A.B, u)
matvecprod_adj(A::MinusLinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = matvecprod_adj(A.A, v)-matvecprod_adj(A.B, v)

## MultLinearOperators: A*B

struct MultLinearOperator{TD,ND,T_,N_,TR,NR}<:AbstractLinearOperator{TD,ND,TR,NR}
    A::AbstractLinearOperator{T_,N_,TR,NR}
    B::AbstractLinearOperator{TD,ND,T_,N_}
end

Base.:*(A::AbstractLinearOperator{T_,N_,TR,NR}, B::AbstractLinearOperator{TD,ND,T_,N_}) where {TD,ND,T_,N_,TR,NR} = MultLinearOperator{TD,ND,T_,N_,TR,NR}(A, B)

domain_size(A::MultLinearOperator) = domain_size(A.B)
range_size(A::MultLinearOperator) = range_size(A.A)
label(A::MultLinearOperator) = string("(", label(A.A), "*", label(A.B),")")
matvecprod(A::MultLinearOperator{TD,ND,T_,N_,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,T_,N_,TR,NR} = matvecprod(A.A, matvecprod(A.B, u))
matvecprod_adj(A::MultLinearOperator{TD,ND,T_,N_,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,T_,N_,TR,NR} = matvecprod_adj(A.B, matvecprod_adj(A.A, v))
invmatvecprod(A::MultLinearOperator{TD,ND,T_,N_,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,T_,N_,TR,NR} = invmatvecprod(A.B, invmatvecprod(A.A, u))
invmatvecprod_adj(A::MultLinearOperator{TD,ND,T_,N_,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,T_,N_,TR,NR} = invmatvecprod_adj(A.A, invmatvecprod_adj(A.B, v))


# Auxiliary operations

## ScaledLinearOperators: c*A

struct ScaledLinearOperator{TD,ND,TR,NR}<:AbstractLinearOperator{TD,ND,TR,NR}
    c::TR
    A::AbstractLinearOperator{TD,ND,TR,NR}
end

Base.:*(c::TR, A::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = ScaledLinearOperator{TD,ND,TR,NR}(c, A)

domain_size(A::ScaledLinearOperator) = domain_size(A.A)
range_size(A::ScaledLinearOperator) = range_size(A.A)
label(A::ScaledLinearOperator) = string("(c*", label(A.A), ")")
matvecprod(A::ScaledLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = A.c*(A.A*u)
matvecprod_adj(A::ScaledLinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = conj(A.c)*matvecprod_adj(A.A, v)
invmatvecprod(A::ScaledLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = invmatvecprod(A.A, u)/A.c
invmatvecprod_adj(A::ScaledLinearOperator{TD,ND,TR,NR}, v::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = invmatvecprod_adj(A.A, v)/conj(A.c)