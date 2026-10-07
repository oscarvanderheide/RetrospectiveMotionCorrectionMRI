using AbstractLinearOperators, LinearAlgebra, CUDA, Test, Random
CUDA.allowscalar(false)
Random.seed!(42)

# Linear operator
input_size = (2^7, 2^8)
padding = (1, 2, 3, 4)
T = Float64
A = zero_padding_operator(T, padding)

# Zero padding test
rtol = T(1e-6)
u = randn(T, input_size)
Au = A*u
@test all(Au[1:padding[1], :] .== 0)
@test all(Au[end-padding[2]+1:end, :] .== 0)
@test all(Au[:, 1:padding[3]] .== 0)
@test all(Au[:, end-padding[4]+1:end] .== 0)
@test all(Au[padding[1]+1:end-padding[2], padding[3]+1:end-padding[4]] .== u)

# Adjoint test
rtol = T(1e-6)
u = randn(T, input_size)
v = randn(T, extended_size(input_size, padding))
@test adjoint_test(A; input=u, output=v, rtol=rtol)

# Full-matrix coherence
u = randn(T, input_size)
Am = full_matrix(A, input_size)
@test vec(A*u) ≈ Am*vec(u) rtol=rtol