using RetrospectiveMotionCorrectionMRI, LinearAlgebra, Test, Random
Random.seed!(123)

T = Float32
rtol = T(1e-5)

@testset "Fused weighted gradient operator" begin
for dim = 1:3, is_complex = [true, false], weighted = [true, false]

    n = tuple(rand(9:14, dim)...)
    h = tuple(abs.(randn(T, dim))...)
    CT = is_complex ? Complex{T} : T

    # Reference: composition of convolution-based gradient and projection
    ∇ = RetrospectiveMotionCorrectionMRI.FastSolversForWeightedTV.gradient_operator(n, h; complex=is_complex)
    P = weighted ? structural_weight(randn(CT, n); η=T(0.1), γ=T(0.9)) : nothing
    A_ref = weighted ? P*∇ : ∇
    A = weighted_gradient_operator(CT, n, h; weight=P)

    u = randn(CT, n)
    p = randn(CT, (n.-1)..., dim)
    @test A*u ≈ A_ref*u rtol=rtol
    @test A'*p ≈ A_ref'*p rtol=rtol
    @test dot(A*u, p) ≈ dot(u, A'*p) rtol=rtol

    # Generic (broadcasting) fallback path, e.g. for non-Array inputs
    @test A*view(u, axes(u)...) ≈ A_ref*u rtol=rtol
    @test A'*view(p, axes(p)...) ≈ A_ref'*p rtol=rtol

    # gradient_norm uses the fused operator, and evaluates as before
    g = gradient_norm(2, 1, n, h; complex=is_complex, weight=P)
    @test g.linear_operator isa WeightedGradientOperator
    @test g(u) ≈ RetrospectiveMotionCorrectionMRI.AbstractProximableFunctions.norm21(A_ref*u) rtol=rtol

end
end
