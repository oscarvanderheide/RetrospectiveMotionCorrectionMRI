using AbstractLinearOperators, LinearAlgebra, CUDA, Test, Random
CUDA.allowscalar(false)
Random.seed!(42)

# Linear operator
input_size = (2^7, 2^8)
output_size = (2^7, 2, 2^7)
T = Float64
A = reshape_operator(T, input_size, output_size)

# Reshape test
rtol = T(1e-6)
u = randn(T, input_size)
@test A*u ≈ reshape(u, output_size) rtol=rtol

# Adjoint test
rtol = T(1e-6)
@test adjoint_test(A; rtol=rtol)