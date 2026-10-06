using RetrospectiveMotionCorrectionMRI, LinearAlgebra, Test, Random
Random.seed!(123)

@testset "Toeplitz normal operator" begin
for T = [Float32, Float64], n = [(16, 14, 12), (15, 13, 11)], motion = [false, true]
    X = spatial_geometry(T.(n.*1e-3), n; origin=T.((1e-3, -2e-3, 0.5e-3)))
    K = kspace_sampling(X, (2, 3)); nt, _ = size(K)
    F = nfft_linop(X, K; tol=T(1e-6))
    θ = motion ? [T(1e-3)*randn(T, nt, 3) T(5e-2)*randn(T, nt, 3)] : zeros(T, nt, 6)
    Fθ = F(θ)
    N = toeplitz_normal_operator(Fθ)
    u = randn(Complex{T}, n); v = randn(Complex{T}, n)
    rtol = T == Float32 ? 1e-4 : 1e-5
    @test N*u ≈ Fθ'*(Fθ*u) rtol=rtol
    @test dot(N*u, v) ≈ dot(u, N*v) rtol=rtol
    @test spectral_radius(N; niter=50) ≈ spectral_radius(Fθ*Fθ'; niter=50) rtol=1e-2
end
end

@testset "Image reconstruction with Toeplitz option" begin
    T = Float32
    n = (24, 20, 16)
    X = spatial_geometry(T.(n.*1e-3), n)
    K = kspace_sampling(X, (2, 3)); nt, _ = size(K)
    F = nfft_linop(X, K; tol=T(1e-6))(T(1e-2)*[1f-1*randn(T, nt, 3) randn(T, nt, 3)])
    u_true = ComplexF32.([((i-12)^2+(j-10)^2+(k-8)^2 < 30) for i=1:n[1], j=1:n[2], k=1:n[3]])
    d = F*u_true .+ 1f-3*randn(ComplexF32, size(F*u_true))
    h = spacing(X)
    g = gradient_norm(2, 1, n, h; complex=true, options=FISTA_options(4f0*sum(1 ./h.^2); niter=20))
    ε = 0.8f0*g(u_true)
    L = 1.1f0*spectral_radius(F*F'; niter=20)
    opt(tp) = image_reconstruction_options(; prox=indicator(g ≤ ε), Lipschitz_constant=L, Nesterov=true, niter=20, fun_history=true, toeplitz=tp)
    o0 = opt(false); o1 = opt(true)
    u0 = image_reconstruction(F, d, zeros(ComplexF32, n), o0)
    u1 = image_reconstruction(F, d, zeros(ComplexF32, n), o1)
    @test u1 ≈ u0 rtol=1e-3
    @test fun_history(o1) ≈ fun_history(o0) rtol=1e-2
    # with Lipschitz estimation
    o2 = image_reconstruction_options(; prox=indicator(g ≤ ε), niter_estimate_Lipschitz=10, Nesterov=true, niter=20, toeplitz=true)
    @test norm(image_reconstruction(F, d, zeros(ComplexF32, n), o2)-u0)/norm(u0) < 1e-2
    # non-NFFT operators are rejected
    @test_throws ArgumentError image_reconstruction(RetrospectiveMotionCorrectionMRI.AbstractLinearOperators.linear_operator(ComplexF32, n, ComplexF32, (prod(n[1:2]), n[3]), u -> reshape(u, :, n[3]), d -> reshape(d, n)), reshape(u_true, :, n[3]), zeros(ComplexF32, n), o1)
end
