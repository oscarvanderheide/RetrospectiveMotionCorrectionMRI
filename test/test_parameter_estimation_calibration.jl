using RetrospectiveMotionCorrectionMRI, LinearAlgebra, Test, Random
Random.seed!(123)

@testset "Parameter estimation with calibration" begin
    n = (32, 32, 32)
    X = spatial_geometry((1f0, 1f0, 1f0), n)
    K = kspace_sampling(X, (1, 2)); nt, _ = size(K)
    F = nfft_linop(X, K)
    # non-separable phantom (with a separable one, in-plane rotations are absorbed by the calibration)
    ground_truth = ComplexF32.([((i-14)^2/40+(j-17)^2/60+(k-15)^2/30 < 1) + 0.5*((i-20)^2+(j-12)^2+(k-20)^2 < 16) for i=1:n[1], j=1:n[2], k=1:n[3]])
    θ_true = zeros(Float32, nt, 6); θ_true[nt÷2+1:end, 5] .= Float32(pi)/180*5
    α = 1 .+ 0.2f0*randn(ComplexF32, nt, 1)            # unknown per-line scaling
    d = α.*(F(θ_true)*ground_truth)
    ti = Float32.(range(1, nt; length=17)); Ip = interpolation1d_motionpars_linop(ti, Float32.(1:nt))
    opt(cal) = parameter_estimation_options(; niter=20, steplength=1f0, scaling_diagonal=1f-3, scaling_mean=1f-3, interp_matrix=Ip, calibration=cal, fun_history=true)
    θ0 = zeros(Float32, length(ti), 6)
    o_cal = opt(true)
    θ_cal = parameter_estimation(F, ground_truth, d, θ0, o_cal)
    @test fun_history(o_cal)[end] < fun_history(o_cal)[1]
    err(θ) = norm(reshape(Ip*vec(θ), nt, 6)-θ_true)/norm(θ_true)
    @test err(θ_cal) < 0.3
end
