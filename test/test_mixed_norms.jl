using LinearAlgebra, AbstractProximableFunctions, Test, Random
Random.seed!(123)

T = Complex{Float64}
n = 32
t = 1e-4
rtol = 10*t

for dim = 1:3, g = [mixed_norm(T,dim,2,2), mixed_norm(T,dim,2,1), mixed_norm(T,dim,2,Inf)]

    # Random data
    y = randn(T, tuple(repeat([n]; outer=dim)...)..., dim)

    # Proxy
    λ = 0.5*norm(y)^2/g(y)
    x = prox(y, λ, g)

    # Gradient test
    fun = prox_objfun(g, λ)
    @test test_grad(fun, y; step=t, rtol=rtol)

    # Projection test
    ε = 0.1*g(y)
    x = proj(y, ε, g)
    @test g(x) ≈ ε rtol=rtol

    ## Gradient test
    fun = proj_objfun(g, ε)
    @test test_grad(fun, y; step=t, rtol=rtol)

end