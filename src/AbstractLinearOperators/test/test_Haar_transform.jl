using AbstractLinearOperators, LinearAlgebra, CUDA, Test, Random
CUDA.allowscalar(false)
Random.seed!(42)

T = Float64
input_size = 2^6
nb = 4
rtol = T(1e-6)

println("Test Haar transform")
for D = 1:3, orthogonal = [true, false], batch = [false, true], device = [:cpu, :gpu]
    println("D=", D, "; orthogonal=", orthogonal, "; batch=", batch, "; device=", device)

    # Linear operator
    W = Haar_transform(T, D; orthogonal=orthogonal, batch=batch)

    # Random input
    sz_u = batch ? (input_size*ones(Integer, D)...,1,nb) : Tuple(input_size*ones(Integer, D))
    sz_v = batch ? (div(input_size,2)*ones(Integer, D)...,2^D,nb) : (div(input_size,2)*ones(Integer, D)...,2^D)
    if device == :cpu
        u = randn(T, sz_u)
        v = randn(T, sz_v)
    else
        u = CUDA.randn(T, sz_u)
        v = CUDA.randn(T, sz_v)
    end

    # Adjoint test
    @test adjoint_test(W; input=u, output=v, rtol=rtol)

    # Inverse test
    @test inverse_test(W; input=u, output=v, rtol=rtol)

end
println()