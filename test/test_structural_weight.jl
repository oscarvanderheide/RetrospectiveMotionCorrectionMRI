using LinearAlgebra, CUDA, FastSolversForWeightedTV, Test, Random
Random.seed!(123)
CUDA.allowscalar(false)

T = Float32
n1d = 16
nc = 3
nb = 16
rtol = T(1e-3)

for dim = 1:3, flag_gpu = [false], is_complex = [true, false]

    # Inputs
    n = tuple(repeat([n1d,]; outer=dim)...)
    h = tuple(abs.(randn(T, dim))...)
    CT = is_complex ? Complex{T} : T

    # Operators
    u = randn(CT, n); flag_gpu && (u = convert(CuArray, u))
    η = structural_mean(u)
    P = structural_weight(u; η=η, γ=T(0.9))

    # Adjoint test
    u = randn(CT, (n.-1)..., dim); flag_gpu && (u = convert(CuArray, u))
    v = randn(CT, (n.-1)..., dim); flag_gpu && (v = convert(CuArray, v))
    @test dot(P*u, v) ≈ dot(u, P'*v) rtol = rtol

end