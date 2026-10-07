using RetrospectiveMotionCorrectionMRI, LinearAlgebra, SparseArrays, Test, Random
Random.seed!(123)

# Reference implementation (as originally in UtilitiesForMRI/src/nfft.jl)
function GN_reference(J; W=nothing, H=nothing)
    WJ = isnothing(W) ? J : cat([W*J[:,:,i] for i = 1:6]...; dims=3)
    HWJ = isnothing(H) ? WJ : cat([H*WJ[:,:,i] for i = 1:6]...; dims=3)
    GN = similar(J, real(eltype(J)), size(J, 1), 6, 6)
    for i = 1:6, j = 1:6
        GN[:,i,j] = vec(real(sum(conj(WJ[:,:,i]).*HWJ[:,:,j]; dims=2)))
    end
    return hvcat(6, [spdiagm(0 => GN[:,i,j]) for j=1:6,i=1:6]...)
end

@testset "Gauss-Newton Hessian and Jacobian adjoint" begin
for T = [Float32, Float64]
    nt, nk = 37, 23
    J = JacobianStructuredNFFTtype2{T}(randn(Complex{T}, nt, nk, 6))
    rtol = T == Float32 ? 1e-5 : 1e-12

    # Hessian (no weights, diagonal weights, non-symmetric pair)
    α = randn(Complex{T}, nt, 1)
    W = RetrospectiveMotionCorrectionMRI.AbstractLinearOperators.linear_operator(Complex{T}, (nt, nk), Complex{T}, (nt, nk), d -> α.*d, d -> conj(α).*d)
    H = RetrospectiveMotionCorrectionMRI.AbstractLinearOperators.linear_operator(Complex{T}, (nt, nk), Complex{T}, (nt, nk), d -> 2 .*d, d -> 2 .*d)
    for (w, h) in [(nothing, nothing), (W, nothing), (W, H), (nothing, H)]
        Hnew = sparse_matrix_GaussNewton(J; W=w, H=h)
        Href = GN_reference(J.∂F; W=w, H=h)
        @test Hnew ≈ Href rtol=rtol
        @test size(Hnew) == (6nt, 6nt)
        @test nnz(Hnew) == 36nt
    end
    @test issymmetric(sparse_matrix_GaussNewton(J))

    # Jacobian adjoint (gradient)
    Δd = randn(Complex{T}, nt, nk)
    @test J'*Δd ≈ real(sum(conj(J.∂F).*Δd; dims=2)[:,1,:]) rtol=rtol
    Δθ = randn(T, nt, 6)
    @test real(dot(J*Δθ, Δd)) ≈ dot(Δθ, J'*Δd) rtol=rtol
end
end
