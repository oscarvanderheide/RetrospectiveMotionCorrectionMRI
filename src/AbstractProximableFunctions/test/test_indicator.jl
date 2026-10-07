using LinearAlgebra, AbstractProximableFunctions, Test, AbstractLinearOperators, Random
Random.seed!(123)

# Random data
T = Complex{Float64}
n = 32
t = 1e-4
rtol = 10*t

for dim = 1:3

    # Random data
    sz = tuple(repeat([n]; outer=dim)...)
    C = zero_set(T, randn(Float64, sz...) .> 0.0)
    δ = indicator(C)
    y = randn(T, sz...)
    v = randn(T, sz..., dim)
    A = linear_operator(T, sz, (sz...,dim), x->v.*x, y->dropdims(sum(conj(v).*y; dims=dim+1); dims=dim+1))
    ρ = 1.01*spectral_radius(A*A'; niter=200)
    opt = FISTA_options(ρ; Nesterov=true, niter=400, reset_counter=20, verbose=false, fun_history=false)

    for g_ = [weighted_prox(mixed_norm(T,dim,2,2), A), weighted_prox(mixed_norm(T,dim,2,1), A), weighted_prox(mixed_norm(T,dim,2,Inf), A)]

        # Proxy
        g = g_+δ
        ε = 0.1*g_(y)

        # Indicator
        h = indicator(g ≤ ε)

        # Prox/Proj
        prox(y, 1.0, h, opt)
        proj(y, 1.0, h, opt)

    end
end