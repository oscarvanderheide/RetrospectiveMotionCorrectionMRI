using LinearAlgebra, RetrospectiveMotionCorrectionMRI, UtilitiesForMRI, Flux, PyPlot, JLD

# Numerical phantom
h = [1f0, 1f0, 1f0]
u_T1 = load("./data/BrainWeb3D_256/BrainWeb3D_256.jld")["T1"]
n = size(u_true)
u_true = complex(u_true/norm(u_true, Inf))

# Fourier operator
X = spatial_sampling(n; h=h)
K = kspace_sampling(X; readout=:z, phase_encode=:xy)
nt, nk = size(K)
F = nfft(X, K; tol=1f-5)
θ_true = imresize(randn(Float32, Int64(nt/128), 6), nt, 6)
smooth_fact = nt/32
w = Kernel.gaussian((smooth_fact, ))
for i = 1:3
    θ_true[:, i] = imfilter(θ_true[:, i], w, "circular")
    θ_true[:, i] .*= 3f0/norm(θ_true[:, i], Inf)
end
for i = 4:6
    θ_true[:, i] = imfilter(θ_true[:, i], w, "circular")
    θ_true[:, i] .*= 2f0/(180f0*norm(θ_true[:, i], Inf))
end

# Data
d = F(θ_true)*u_true

# Optimization options
niter = 200
steplength = 1f0
λ = 0f0
cdiag = 1f-5
cid = 1f-1
hist_size = 10
β = 1f0
ti = Float32.(range(1, nt; length=Int64(nt/16)))
t = Float32.(1:nt)
Ip = interpolation1d_motionpars_linop(ti, t)
D = derivative1d_motionpars_linop(t, 2; pars=(true, true, true, true, true, true))/4f0
opt = parameter_estimation_options(; niter=niter, steplength=steplength, λ=λ, cdiag=cdiag, cid=cid,  reg_matrix=D, interp_matrix=Ip, verbose=true)

# Solution
θ0 = zeros(Float32, length(ti)*6)
θ_sol, fval = parameter_estimation(F, u_true, d, θ0, opt)
θ_sol_p = reshape(Ip*vec(θ_sol), length(t), 6)
θ_sol_p = reshape(θ_sol_p, n[1:2]..., 6)