using AbstractLinearOperators, LinearAlgebra, CUDA, Test, Random
CUDA.allowscalar(false)
Random.seed!(42)

# Custom-type module
module ModuleCustomType
    using AbstractLinearOperators
    import AbstractLinearOperators: domain_size, range_size, matvecprod, matvecprod_adj, invmatvecprod, invmatvecprod_adj

    export CustomType

    struct CustomType{T,ND,NR}<:AbstractLinearOperator{T,ND,T,NR}
        dsize::NTuple{ND,Integer}
        rsize::NTuple{NR,Integer}
        A::Array{T,2}
    end

    domain_size(A::CustomType) = A.dsize
    range_size(A::CustomType) = A.rsize
    matvecprod(A::CustomType{T,ND,NR}, u::AbstractArray{T,ND}) where {T,ND,NR} = reshape(A.A*reshape(u, A.dsize[1]*A.dsize[2], A.dsize[3]*A.dsize[4]), A.rsize)
    matvecprod_adj(A::CustomType{T,ND,NR}, v::AbstractArray{T,NR}) where {T,ND,NR} = reshape(adjoint(A.A)*reshape(v, A.rsize[1]*A.rsize[2], A.rsize[3]*A.rsize[4]), A.dsize)
    invmatvecprod(A::CustomType{T,ND,NR}, v::AbstractArray{T,NR}) where {T,ND,NR} = reshape(A.A\reshape(v, A.rsize[1]*A.rsize[2], A.rsize[3]*A.rsize[4]), A.dsize)
    invmatvecprod_adj(A::CustomType{T,ND,NR}, u::AbstractArray{T,ND}) where {T,ND,NR} = reshape(adjoint(A.A)\reshape(u, A.rsize[1]*A.rsize[2], A.rsize[3]*A.rsize[4]), A.rsize)
end
using .ModuleCustomType

# Precision
T = Float64
rtol = T(1e-6)

# Random init
dsize = (2, 3, 4, 5)
rsize = (6, 7, 10, 2)
A = CustomType{T,4,4}(dsize, rsize, randn(T, 6*7, 6))
B = CustomType{T,4,4}(dsize, rsize, randn(T, 6*7, 6))
u = randn(T, dsize)
v = randn(T, rsize)

# Adjoint
C = adjoint(A)
@test C*v ≈ A'*v rtol=rtol
@test adjoint_test(C; rtol=rtol)

# -
C = -A
@test C*u ≈ -(A*u) rtol=rtol
@test adjoint_test(C; rtol=rtol)

# +/-
C = A+B
@test C*u ≈ A*u+B*u rtol=rtol
@test adjoint_test(C; rtol=rtol)
C = A-B
@test C*u ≈ A*u-B*u rtol=rtol
@test adjoint_test(C; rtol=rtol)

# *
rsize2 = (7, 2, 5, 4)
v = randn(T, rsize2)
B = CustomType{T,4,4}(rsize, rsize2, randn(T,7*2,42))
C = B*A
@test C*u ≈ B*(A*u) rtol=rtol
@test adjoint_test(C; rtol=rtol)

# c*
α = randn(T)
C = α*A
@test C*u ≈ α*(A*u) rtol=rtol
@test adjoint_test(C; rtol=rtol)


# Random init
dsize = (6, 7, 2, 21)
A = CustomType{T,4,4}(dsize, dsize, randn(T, 42, 42))
B = CustomType{T,4,4}(dsize, dsize, randn(T, 42, 42))
u = randn(T, dsize)
v = randn(T, dsize)

# Inverse
C = inv(A)
@test C*u ≈ A\u rtol=rtol
@test adjoint_test(C; rtol=rtol)
@test inverse_test(C; rtol=rtol)

# Adjoint
C = adjoint(A)
@test C*v ≈ A'*v rtol=rtol
@test adjoint_test(C; rtol=rtol)
@test inverse_test(C; rtol=rtol)

# -
C = -A
@test C*u ≈ -(A*u) rtol=rtol
@test adjoint_test(C; rtol=rtol)
@test inverse_test(C; rtol=rtol)

# *
C = B*A
@test C*u ≈ B*(A*u) rtol=rtol
@test adjoint_test(C; rtol=rtol)
@test inverse_test(C; rtol=rtol)

# c*
α = randn(T)
C = α*A
@test C*u ≈ α*(A*u) rtol=rtol
@test adjoint_test(C; rtol=rtol)
@test inverse_test(C; rtol=rtol)