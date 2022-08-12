using LinearAlgebra, RetrospectiveMotionCorrectionMRI, Test

# Input
d0 = randn(ComplexF64, 10^2, 10)
p = randn(ComplexF64, 10^2, 10); p .*= norm(d0)/norm(p)
t = 1e-6

# Loss functions
for loss in [data_residual_loss(ComplexF64, 2, 2), data_residual_loss(ComplexF64, 2, 1), data_residual_loss(ComplexF64, 2, 1; ρ=1.0)]

    # Gradient test
    f0, g0, H0 = loss(d0; Hessian=true)
    fp1, gp1 = loss(d0+t/2*p)
    fm1, gm1 = loss(d0-t/2*p)
    @test (fp1-fm1)/t ≈ real(dot(g0, p)) rtol=t*1e1
    @test (gp1-gm1)/t ≈ H0*p rtol=t*1e2

end