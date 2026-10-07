using AbstractLinearOperators, LinearAlgebra, CUDA, Test, Random
CUDA.allowscalar(false)
Random.seed!(42)

# Linear operator
input_size = (2^7, 2^8)
padding = (1, 2, 3, 4)
T = Float64
A = repeat_padding_operator(T, padding)

# Repeat padding test
rtol = T(1e-6)
u = randn(T, input_size)
Au = A*u
@test all(Au[padding[1]+1:end-padding[2], padding[3]+1:end-padding[4]] .== u)

# Adjoint test
rtol = T(1e-6)
u = randn(T, input_size)
v = randn(T, extended_size(input_size, padding))
@test adjoint_test(A; input=u, output=v, rtol=rtol)