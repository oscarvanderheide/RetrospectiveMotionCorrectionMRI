using LinearAlgebra, RetrospectiveMotionCorrectionMRI.AbstractProximableFunctions, Test, RetrospectiveMotionCorrectionMRI.AbstractLinearOperators, Random
Random.seed!(123)

# Random data
T = Complex{Float64}
n = 16
t = 1e-4
rtol = 10*t

for dim = 1:3

    # Random data
    sz = tuple(repeat([n]; outer=dim)...)
    C = zero_set(T, randn(Float64, sz...) .> 0.0)
    δ = indicator(C)
    y = randn(T, sz...)
    v = randn(T, sz..., dim)
    A = linear_operator(T, sz, T, (sz...,dim), x->v.*x, y->dropdims(sum(conj(v).*y; dims=dim+1); dims=dim+1))
    ρ = 1.01*spectral_radius(A*A'; niter=200)
    opt = FISTA_options(ρ; Nesterov=true, niter=400, reset_counter=20, verbose=false, fun_history=false)

    for g_ = [weighted_prox(mixed_norm(T,dim,2,2), A), weighted_prox(mixed_norm(T,dim,2,1), A), weighted_prox(mixed_norm(T,dim,2,Inf), A)]

        # Proxy
        g = g_+δ
        λ = 0.5*norm(y)^2/g_(y)
        x = prox(y, λ, g, opt)

        # Projection test
        @test x ∈ C

        # Gradient test (prox)
        fun = prox_objfun(g, λ; options=opt)
        @test test_grad(fun, y; step=t, rtol=rtol)

        # Projection test
        ε = 0.1*g_(y)
        x = proj(y, ε, g, opt)
        @test (g(x) ≤ ε) || (abs(g(x)-ε) ≤ rtol*ε)
        @test x ∈ C

        ## Gradient test (projion)
        fun = proj_objfun(g, ε; options=opt)
        # (the projection is computed with a finite number of inner FISTA iterations; the finite-difference
        # check of its gradient is only as accurate as that inner solve, hence the looser tolerance)
        @test test_grad(fun, y; step=t, rtol=2e-2)

    end
end