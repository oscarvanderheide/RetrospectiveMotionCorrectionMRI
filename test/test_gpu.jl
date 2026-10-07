using CUDA, NonuniformFFTs, RetrospectiveMotionCorrectionMRI, LinearAlgebra, Test, Random
const APF = RetrospectiveMotionCorrectionMRI.AbstractProximableFunctions
CUDA.allowscalar(false)
Random.seed!(123)

# GPU (CUDA extension) vs CPU implementation
@testset "GPU" begin

    rel(a, b) = norm(Array(a)-Array(b))/norm(Array(b))
    @test Base.get_extension(RetrospectiveMotionCorrectionMRI, :RetrospectiveMotionCorrectionMRICUDAExt) !== nothing

    n = (24, 20, 18)
    X = spatial_geometry(n.*1f-3, n; origin=(1f-3, -2f-3, 0.5f-3))
    K = kspace_sampling(X, (2, 3)); nt, _ = size(K)
    F = nfft_linop(X, K; tol=1f-6); Fg = cu(F)
    θ = [1f-3*randn(Float32, nt, 3) 3f-2*randn(Float32, nt, 3)]
    u = randn(ComplexF32, n); d = randn(ComplexF32, nt, n[1])
    tol = 5f-5 # GPU and CPU NUFFTs both have errors ~1e-5 at tol=1e-6

    @testset "Point folding (NonuniformFFTs edge case)" begin
        # Tiny negative coordinates (e.g. rotated k = 0) must be folded inside [0, 2π), not onto 2π
        ext = Base.get_extension(RetrospectiveMotionCorrectionMRI, :RetrospectiveMotionCorrectionMRICUDAExt)
        xs = cu([-1f-8, -1f-7, -0f0, 0f0, Float32(2π), -Float32(2π), 1f0, -1f0, 10f0, -10f0])
        ys = Array(ext.fold_unit_cell.(xs))
        @test all(y -> 0 <= y < Float32(2π), ys)
        @test ys[7] == 1f0
        # NUFFT with exactly zero (and tiny negative) coordinates matches the CPU
        θ0 = zeros(Float32, nt, 6); θ0[:, 4] .= 1f-9
        @test rel(Fg(θ0)*cu(u), F(θ0)*u) < tol
        @test rel(Fg(θ0)'*cu(d), F(θ0)'*d) < tol
    end

    @testset "NUFFT and rigid motion" begin
        @test Fg.kcoord isa CuArray
        @test rel(Fg*cu(u), F*u) < tol
        @test rel(Fg'*cu(d), F'*d) < tol
        @test rel(Fg(θ)*cu(u), F(θ)*u) < tol
        @test rel(Fg(θ)'*cu(d), F(θ)'*d) < tol
        @test real(dot(Fg(θ)*cu(u), cu(d))) ≈ real(dot(cu(u), Fg(θ)'*cu(d))) rtol=1e-4
    end

    @testset "Jacobian and Gauss-Newton" begin
        dg, _, Jg = ∂(Fg()*cu(u), θ); dc, _, Jc = ∂(F()*u, θ)
        @test rel(dg, dc) < tol
        @test rel(Jg.∂F, Jc.∂F) < tol
        @test rel(Jg'*cu(d), Jc'*d) < tol
        Δθ = randn(Float32, nt, 6)
        @test rel(Jg*Δθ, Jc*Δθ) < tol
        Hg = sparse_matrix_GaussNewton(Jg); Hc = sparse_matrix_GaussNewton(Jc)
        @test norm(Array(Hg-Hc))/norm(Array(Hc)) < tol
    end

    @testset "Weighted TV" begin
        h = spacing(X)
        ref = abs.(u) .+ 0f0im
        P = structural_weight(ref; η=0.1f0); Pg = structural_weight(cu(ref); η=0.1f0)
        @test Pg.ξ isa CuArray
        @test rel(Pg.ξ, P.ξ) < 1e-5
        for warmstart in (false, true)
            opt = FISTA_options(4f0*sum(1 ./h.^2); niter=10)
            g = gradient_norm(2, 1, n, h; complex=true, weight=P, options=opt, warmstart=warmstart)
            gg = gradient_norm(2, 1, n, h; complex=true, weight=Pg, options=opt, warmstart=warmstart)
            p = randn(ComplexF32, (n.-1)..., 3)
            @test rel(gg.linear_operator*cu(u), g.linear_operator*u) < 1e-5
            @test rel(gg.linear_operator'*cu(p), g.linear_operator'*p) < 1e-5
            ε = 0.5f0*g(u)
            for _ = 1:2 # (second call exercises the warm start)
                @test rel(APF.proj!(cu(u), ε, gg, opt, similar(cu(u))), APF.proj!(u, ε, g, opt, similar(u))) < 1e-5
            end
        end
    end

    @testset "Toeplitz normal operator" begin
        Ng = toeplitz_normal_operator(Fg(θ))
        @test Ng.λ isa CuArray
        @test rel(Ng*cu(u), F(θ)'*(F(θ)*u)) < tol
    end

    @testset "Image reconstruction, parameter estimation, motion correction" begin
        img = ComplexF32.([((i-12)^2/30+(j-10)^2/20+(k-9)^2/15 < 1) for i=1:n[1], j=1:n[2], k=1:n[3]])
        θt = zeros(Float32, nt, 6); θt[nt÷2:end, 6] .= 0.03f0
        dt = F(θt)*img
        h = spacing(X)
        mk(r) = gradient_norm(2, 1, n, h; complex=true, weight=structural_weight(r; η=1f-2), options=FISTA_options(4f0*sum(1 ./h.^2); niter=10))
        g = mk(img); gg = mk(cu(img)); ε = g(img)
        for toeplitz in (false, true)
            oi(gx) = image_reconstruction_options(; prox=indicator(gx ≤ ε), niter=5, niter_estimate_Lipschitz=5, toeplitz=toeplitz)
            Random.seed!(1); uc = image_reconstruction(F, dt, zeros(ComplexF32, n), oi(g))
            Random.seed!(1); ug = image_reconstruction(Fg, cu(dt), cu(zeros(ComplexF32, n)), oi(gg))
            @test ug isa CuArray
            @test rel(ug, uc) < 1e-4
        end
        ti = Float32.(range(1, nt; length=8)); Ip = interpolation1d_motionpars_linop(ti, Float32.(1:nt))
        op = parameter_estimation_options(; niter=3, steplength=1f0, scaling_diagonal=1f-3, scaling_mean=1f-4, interp_matrix=Ip)
        θc = parameter_estimation(F, img, dt, zeros(Float32, 8, 6), op)
        θg = parameter_estimation(Fg, cu(img), cu(dt), zeros(Float32, 8, 6), op)
        @test θg isa Matrix{Float32}
        @test rel(θg, θc) < 1e-3
        oi = image_reconstruction_options(; prox=indicator(gg ≤ ε), niter=3, niter_estimate_Lipschitz=3)
        om = motion_correction_options(; image_reconstruction_options=oi, parameter_estimation_options=op, niter=2)
        ug, θg = motion_corrected_reconstruction(Fg, cu(dt), cu(zeros(ComplexF32, n)), zeros(Float32, 8, 6), om)
        @test ug isa CuArray && θg isa Matrix{Float32}
        @test all(isfinite, Array(ug))
    end

end
