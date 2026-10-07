using RetrospectiveMotionCorrectionMRI, LinearAlgebra, Test, Random
Random.seed!(123)

@testset "NFFT Jacobian" begin
for T = [Float32, Float64]
    n = (20, 18, 16)
    X = spatial_geometry(T.(n.*1e-3), n; origin=T.((0.5e-3, -1e-3, 2e-3)))
    K = kspace_sampling(X, (2, 3)); nt, _ = size(K)
    F = nfft_linop(X, K; tol=T(1e-6))
    θ = [T(1e-3)*randn(T, nt, 3) T(2e-2)*randn(T, nt, 3)]
    u = randn(Complex{T}, n)

    d, Fθ, J = ∂(F()*u, θ)
    rtol = T == Float32 ? 1e-4 : 1e-8

    # Reference: explicit composition of rotation, NUFFT and phase-shift derivatives
    Kφ, ∂Kφ = ∂(rotation()()*F.kcoord, θ[:,4:6])
    Fφ = nfft_linop(X, Kφ; tol=F.tol, norm_constant=F.norm_constant)
    x, y, z = coord(X; mesh=false)
    ∇Fu = -im*cat(Fφ*(u.*reshape(x,:,1,1)), Fφ*(u.*reshape(y,1,:,1)), Fφ*(u.*reshape(z,1,1,:)); dims=3)
    d_ref, Pτ, ∂Pτu = ∂(phase_shift(F.kcoord)()*(Fφ*u), θ[:,1:3])
    J_ref = cat(∂Pτu.∂P, Pτ*dot(∇Fu, ∂Kφ); dims=3)

    @test d ≈ F(θ)*u rtol=rtol
    @test d ≈ d_ref rtol=rtol
    @test J.∂F ≈ J_ref rtol=rtol
    @test Fθ*u ≈ d rtol=rtol

    # Finite-difference check of the Jacobian
    if T == Float64
        Δθ = randn(T, nt, 6).*[fill(T(1e-3), 1, 3) fill(T(1e-2), 1, 3)]
        t = T(1e-6)
        fd = (F(θ+t/2*Δθ)*u-F(θ-t/2*Δθ)*u)/t
        @test fd ≈ J*Δθ rtol=1e-4
    end
end
end
