using LinearAlgebra, FastSolversForWeightedTV, AbstractProximableFunctions, CUDA, Test, Random
Random.seed!(123)
CUDA.allowscalar(false)

T = Float64
n1d = 16
nc = 3
step = T(1e-5)
rtol = T(1e-3)

for dim = 1:3, flag_gpu = [false], is_complex = [true, false]
    
    # FISTA optimizer
    niter = 1000
    opt = FISTA_options(T(4*dim); Nesterov=true, niter=niter, reset_counter=20, verbose=false)

    # TV norm
    CT = is_complex ? Complex{T} : T
    n = tuple(repeat([n1d,]; outer=dim)...)
    h = tuple(ones(T, dim)...)
    y = randn(CT, n); flag_gpu && (y = convert(CuArray, y))
    g = gradient_norm(2, 1, n, h; complex=is_complex)

    # Constraint
    is_zero = randn(T, size(y)).>0; flag_gpu && (is_zero = convert(CuArray, is_zero))
    C = zero_set(CT, is_zero)

    # Gradient test (proxy)
    λ = T(0.5)*norm(y)^2/g(y)
    fun = prox_objfun(g, λ; options=opt)
    @test test_grad(fun, y; step=step, rtol=rtol)

    # Projection test
    ε = T(0.1)*g(y)
    x = proj(y, ε, g, opt)
    @test (g(x) ≤ ε) || isapprox(g(x), ε; rtol=rtol)

    # Gradient test (projection)
    fun = proj_objfun(g, ε; options=opt)
    @test test_grad(fun, y; step=step, rtol=rtol)

end