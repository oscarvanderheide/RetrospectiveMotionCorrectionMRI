using RetrospectiveMotionCorrectionMRI, LinearAlgebra, Test, Random
const APF = RetrospectiveMotionCorrectionMRI.AbstractProximableFunctions
const FSW = RetrospectiveMotionCorrectionMRI.FastSolversForWeightedTV
Random.seed!(123)

@testset "Fast weighted TV-ball projection" begin

    # Exact L1-ball threshold vs Brent root search
    for T = [Float32, Float64], N = [10, 1000, 100_000]
        a = abs.(randn(T, N)).^2
        ε = T(0.3)*sum(a)
        λ = FSW.l1ball_threshold(a, ε)
        @test sum(max.(a.-λ, 0)) ≈ ε rtol=sqrt(eps(T))
        @test λ ≈ APF.pareto_search_projL1(a, ε) rtol=sqrt(eps(T))
        @test FSW.l1ball_threshold(a, 2*sum(a)) == 0
    end

    # Specialized solver vs generic dual FISTA
    for T = [Float32, Float64], is_complex = [true, false], weighted = [true, false], (Nesterov, reset_counter) = [(true, nothing), (true, 3), (false, nothing)]
        n = (13, 11, 9); h = (T(1), T(0.8), T(1.3))
        CT = is_complex ? Complex{T} : T
        ref = randn(CT, n)
        P = weighted ? structural_weight(ref; η=T(0.1)) : nothing
        opt = FISTA_options(4*sum(1 ./h.^2); Nesterov=Nesterov, reset_counter=reset_counter, niter=20, fun_history=true)
        g = gradient_norm(2, 1, n, h; complex=is_complex, weight=P, options=opt)
        y = randn(CT, n)
        ε = T(0.2)*g(y)
        x_fast = APF.proj!(y, ε, g, opt, similar(y))
        hist_fast = copy(fun_history(opt))
        x_gen = APF.proj_weighted_dual_fista!(y, ε, g, opt, similar(y))
        hist_gen = copy(fun_history(opt))
        tol = T == Float32 ? 1e-4 : 1e-10
        @test x_fast ≈ x_gen rtol=tol
        @test hist_fast ≈ hist_gen rtol=tol
        # inactive constraint: projection is the identity
        @test APF.proj!(y, 2*g(y), g, opt, similar(y)) ≈ y rtol=tol
    end

    # Non-3D, non-Array, or non-fused operators still use the generic implementation
    n = (12, 10); h = (1f0, 1f0)
    g = gradient_norm(2, 1, n, h; complex=false, options=FISTA_options(8f0; niter=5))
    y = randn(Float32, n)
    @test APF.proj!(y, 0.5f0*g(y), g, g.options, similar(y)) isa Matrix{Float32}

end
