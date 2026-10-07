using LinearAlgebra, CUDA, FastSolversForWeightedTV, Test, Random
Random.seed!(123)
CUDA.allowscalar(false)

T = Float32
n1d = 16
nc = 3
nb = 16
rtol = T(1e-3)

for dim = 1:3, flag_gpu = [false], is_complex = [true, false]

    # Operator
    n = tuple(repeat([n1d,]; outer=dim)...)
    h = tuple(abs.(randn(T, dim))...)
    ∇ = gradient_operator(n, h; complex=is_complex)

    # Adjoint test
    CT = is_complex ? Complex{T} : T
    u = randn(CT, n);              flag_gpu && (u = convert(CuArray, u))
    v = randn(CT, (n.-1)..., dim); flag_gpu && (v = convert(CuArray, v))
    @test dot(∇*u, v) ≈ dot(u, ∇'*v) rtol=rtol

    # Consistency w/ stencil-free gradient eval
    @test ∇*u ≈ gradient_eval(u, h) rtol=rtol

end