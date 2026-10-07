using AbstractLinearOperators, LinearAlgebra, CUDA, Test, Random
CUDA.allowscalar(false)
Random.seed!(42)

# Linear operator
input_size = (2^7, 2^8)
T = ComplexF64
Id = identity_operator(T, 2)

# Identity test
rtol = real(T)(1e-6)
u = randn(T, input_size)
@test Id*u ≈ u rtol=rtol

# Adjoint test
u = randn(T, input_size)
v = randn(T, input_size)
rtol = real(T)(1e-6)
@test adjoint_test(Id; input=u, output=v, rtol=rtol)

# Full matrix coherence
Idmat = full_matrix(Id, input_size)
@test vec(Id*u) ≈ Idmat*vec(u) rtol=rtol