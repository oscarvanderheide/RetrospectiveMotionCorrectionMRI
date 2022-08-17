using RetrospectiveMotionCorrectionMRI, FastSolversForWeightedTV, UtilitiesForMRI, ConvexOptimizationUtils, LinearAlgebra, Test

# Spatial geometry
fov = (1f0, 2f0, 2f0)
n = (64, 64, 64)
o = (0.5f0, 1f0, 1f0)
X = spatial_geometry(fov, n; origin=o)

# Cartesian sampling in k-space
phase_encoding = (1,2)
K = kspace_sampling(X, phase_encoding)
nt, nk = size(K)

# Fourier operator
F = nfft_linop(X, K)

# Setting ground-truth and data
ground_truth = zeros(ComplexF32, n); ground_truth[div(64,2)+1-10:div(64,2)+1+10, div(64,2)+1-10:div(64,2)+1+10, div(64,2)+1-10:div(64,2)+1+10] .= 1
nt, _ = size(K)
θ_true = zeros(Float32, nt, 6)
θ_true[div(nt,2)+51:end,:] .= reshape([0.0f0, 0.0f0, 0.0f0, Float32(pi)/180*10, 0f0, 0f0], 1, 6)
d = F(θ_true)*ground_truth
u_conventional = F'*d

# Optimization options
ti = Float32.(range(1, nt; length=16))
# ti = Float32.(1:nt)
t = Float32.(1:nt)
Ip = interpolation1d_motionpars_linop(ti, t)
D = derivative1d_motionpars_linop(t, 2; pars=(true, true, true, true, true, true))/4f0
opt = parameter_estimation_options(; niter=10,
                                     steplength=1f0,
                                     λ=0f0,
                                     cdiag=1f-5, cid=1f-1,
                                     reg_matrix=D,
                                     interp_matrix=Ip,
                                     verbose=true,
                                     fun_history=true)

# Solution
θ0 = zeros(Float32, length(ti)*6)
θ_sol = parameter_estimation(F, ground_truth, d, θ0, opt)
θ_sol_p = reshape(Ip*vec(θ_sol), length(t), 6)