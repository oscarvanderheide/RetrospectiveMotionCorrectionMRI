using RetrospectiveMotionCorrectionMRI, LinearAlgebra, Test, Random
const APF = RetrospectiveMotionCorrectionMRI.AbstractProximableFunctions
const ALO = RetrospectiveMotionCorrectionMRI.AbstractLinearOperators
Random.seed!(123)

# Linear operator counting forward evaluations
mutable struct CountingOp{T}<:ALO.AbstractLinearOperator{T,1,T,1}
    M::Matrix{T}
    nfwd::Int
end
ALO.domain_size(A::CountingOp) = (size(A.M, 2),)
ALO.range_size(A::CountingOp) = (size(A.M, 1),)
ALO.matvecprod(A::CountingOp{T}, u::AbstractArray{T,1}) where T = (A.nfwd += 1; A.M*u)
ALO.matvecprod_adj(A::CountingOp{T}, v::AbstractArray{T,1}) where T = A.M'*v

@testset "Least-squares misfit" begin
    A = CountingOp(randn(ComplexF64, 20, 10), 0)
    y = randn(ComplexF64, 20); x = randn(ComplexF64, 10)
    f = leastsquares_misfit(A, y)
    g = similar(x)
    fval = APF.fungradeval!(f, x, g)
    @test fval ≈ norm(A.M*x-y)^2/2
    @test g ≈ A.M'*(A.M*x-y)
    A.nfwd = 0; APF.fungradeval!(f, x, g)
    @test A.nfwd == 1 # one forward evaluation per function+gradient evaluation
end
